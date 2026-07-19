using System.Text.Json;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Runs;

if (args.Length == 5 && args[0] == "--rng-probe")
{
    string runSeed = args[1];
    int counter = int.Parse(args[2]);
    int maximum = int.Parse(args[3]);
    int count = int.Parse(args[4]);
    var rng = new RunRngSet(runSeed).CombatTargets;
    rng.FastForwardCounter(counter);
    var values = Enumerable.Range(0, count).Select(_ => rng.NextInt(maximum)).ToList();
    Console.WriteLine(JsonSerializer.Serialize(new Dictionary<string, object?>
    {
        ["run_seed"] = runSeed,
        ["initial_counter"] = counter,
        ["maximum"] = maximum,
        ["values"] = values,
        ["final_counter"] = rng.Counter,
        ["stream_seed"] = rng.Seed
    }, new JsonSerializerOptions { WriteIndented = true }));
    return 0;
}

if (args.Length is < 1 or > 4)
{
    Console.Error.WriteLine(
        "Usage: ReplayInspector <replay.mcr> [--full --expected-version <version>]\n"
        + "       ReplayInspector --rng-probe <run-seed> <counter> <maximum> <count>");
    return 2;
}

string input = Path.GetFullPath(args[0]);
bool readFullReplay = args.Contains("--full", StringComparer.Ordinal);
int expectedIndex = Array.IndexOf(args, "--expected-version");
string? expectedVersion = expectedIndex >= 0 && expectedIndex + 1 < args.Length
    ? args[expectedIndex + 1]
    : null;
if (readFullReplay && expectedVersion == null)
{
    Console.Error.WriteLine("--full requires --expected-version to prevent cross-patch decoding.");
    return 2;
}
if (!File.Exists(input))
{
    Console.Error.WriteLine($"Replay file not found: {input}");
    return 2;
}

try
{
    byte[] bytes = File.ReadAllBytes(input);
    var headerReader = new PacketReader();
    headerReader.Reset(bytes);
    string headerVersion = headerReader.ReadString();
    string headerCommit = headerReader.ReadString();
    uint headerModelIdHash = headerReader.ReadUInt();
    var header = new Dictionary<string, object?>
    {
        ["schema_version"] = 1,
        ["format"] = "sts2_combat_replay_header",
        ["version"] = headerVersion,
        ["commit"] = headerCommit,
        ["model_id_hash"] = headerModelIdHash,
        ["bytes"] = bytes.Length
    };
    if (!readFullReplay)
    {
        Console.WriteLine(JsonSerializer.Serialize(header, new JsonSerializerOptions
        {
            WriteIndented = true
        }));
        return 0;
    }
    if (!string.Equals(headerVersion, expectedVersion, StringComparison.Ordinal))
    {
        Console.Error.WriteLine(
            $"Replay is {headerVersion}, not expected {expectedVersion}; refusing full decode.");
        return 3;
    }

    uint loadedModelHash = ReplayCacheInitializer.InitializeBaseGame();
    if (loadedModelHash != headerModelIdHash)
    {
        Console.Error.WriteLine(
            $"Replay model table hash is {headerModelIdHash}, but the loaded base game table is "
            + $"{loadedModelHash}. Load the matching mods or use a vanilla replay.");
        return 4;
    }

    var reader = new PacketReader();
    reader.Reset(bytes);
    CombatReplay replay = reader.Read<CombatReplay>();
    var players = replay.serializableRun.Players;
    var playerIndexes = players
        .Select((player, index) => (player.NetId, index))
        .ToDictionary(item => item.NetId, item => item.index);
    var actions = new List<Dictionary<string, object?>>();
    var eventCounts = new Dictionary<string, int>(StringComparer.Ordinal);
    var actionCounts = new Dictionary<string, int>(StringComparer.Ordinal);

    for (int replayEventIndex = 0; replayEventIndex < replay.events.Count; replayEventIndex++)
    {
        CombatReplayEvent replayEvent = replay.events[replayEventIndex];
        string eventType = replayEvent.eventType.ToString();
        eventCounts[eventType] = eventCounts.GetValueOrDefault(eventType) + 1;
        if (replayEvent.action == null)
            continue;

        string actionType = replayEvent.action.GetType().Name;
        actionCounts[actionType] = actionCounts.GetValueOrDefault(actionType) + 1;
        var action = new Dictionary<string, object?>
        {
            ["replay_event_index"] = replayEventIndex,
            ["action_type"] = actionType,
            ["player_index"] = replayEvent.playerId.HasValue
                && playerIndexes.TryGetValue(replayEvent.playerId.Value, out int playerIndex)
                    ? playerIndex
                    : null
        };
        switch (replayEvent.action)
        {
            case NetPlayCardAction play:
                action["card_id"] = play.modelId.Entry;
                action["combat_card_index"] = play.card.CombatCardIndex;
                action["target_combat_id"] = play.targetId;
                break;
            case NetUsePotionAction potion:
                action["potion_slot"] = potion.potionIndex;
                action["target_combat_id"] = potion.targetId;
                action["target_player_index"] = potion.targetPlayerId.HasValue
                    && playerIndexes.TryGetValue(potion.targetPlayerId.Value, out int targetPlayerIndex)
                        ? targetPlayerIndex
                        : null;
                action["in_combat"] = potion.enqueuedInCombat;
                break;
            case NetEndPlayerTurnAction endTurn:
                action["turn_number"] = endTurn.turnNumber;
                break;
        }
        actions.Add(action);
    }

    var result = new Dictionary<string, object?>
    {
        ["schema_version"] = 1,
        ["format"] = "sts2_combat_replay_summary",
        ["replay"] = new Dictionary<string, object?>
        {
            ["version"] = replay.version,
            ["commit"] = replay.gitCommit,
            ["model_id_hash"] = replay.modelIdHash,
            ["ascension"] = replay.serializableRun.Ascension,
            ["act"] = replay.serializableRun.CurrentActIndex + 1,
            ["floor"] = replay.serializableRun.FloorReached,
            ["players"] = players.Select(player => new Dictionary<string, object?>
            {
                ["character"] = player.CharacterId?.Entry,
                ["current_hp"] = player.CurrentHp,
                ["max_hp"] = player.MaxHp,
                ["deck_size"] = player.Deck.Count,
                ["relic_count"] = player.Relics.Count,
                ["potion_count"] = player.Potions.Count
            }).ToList(),
            ["events"] = replay.events.Count,
            ["event_counts"] = eventCounts,
            ["action_counts"] = actionCounts,
            ["checksums"] = replay.checksumData.Count,
            ["checksum_contexts"] = replay.checksumData
                .Select(checksum => checksum.context)
                .Distinct()
                .Order()
                .ToList()
        },
        ["actions"] = actions
    };
    Console.WriteLine(JsonSerializer.Serialize(result, new JsonSerializerOptions
    {
        WriteIndented = true
    }));
    return 0;
}
catch (Exception exception)
{
    Console.Error.WriteLine($"Could not read replay: {exception}");
    return 1;
}
