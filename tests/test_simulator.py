from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stsai.simulator import (
    CardCatalog,
    CardInstance,
    CombatState,
    CreatureState,
    PlayerPhaseSimulator,
    StsRngState,
    UnsupportedMechanic,
    audit_replay_coverage,
)


CATALOG = (
    Path(__file__).parents[1]
    / "data"
    / "game"
    / "v0.107.1"
    / "silent-starter-cards.json"
)


def _state(*cards: CardInstance, energy: int = 3) -> CombatState:
    return CombatState(
        patch="v0.107.1",
        character="CHARACTER.SILENT",
        turn=1,
        energy=energy,
        max_energy=3,
        player=CreatureState("PLAYER", 70, 70),
        enemies=(
            CreatureState("CULTIST_0", 20, 20),
            CreatureState("CULTIST_1", 20, 20),
        ),
        hand=tuple(cards),
    )


class SimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = CardCatalog.from_path(CATALOG)
        self.simulator = PlayerPhaseSimulator(self.catalog)

    def test_legal_mask_is_character_conditioned_and_expands_targets(self) -> None:
        state = _state(
            CardInstance(1, "CARD.STRIKE_SILENT"),
            CardInstance(2, "CARD.NEUTRALIZE"),
            energy=0,
        )
        actions = self.simulator.legal_actions(state)
        self.assertNotIn({"action": "play_card", "card_index": 0}, actions)
        self.assertIn(
            {"action": "play_card", "card_index": 1, "target": "CULTIST_0"},
            actions,
        )
        wrong_character = CombatState(**{**state.__dict__, "character": "IRONCLAD"})
        with self.assertRaises(ValueError):
            self.simulator.legal_actions(wrong_character)

    def test_strike_neutralize_and_defend_are_immutable(self) -> None:
        state = _state(
            CardInstance(1, "STRIKE_SILENT"),
            CardInstance(2, "NEUTRALIZE", upgraded=True),
            CardInstance(3, "DEFEND_SILENT"),
        )
        strike = self.simulator.step(
            state, {"action": "play_card", "card_index": 0, "target": "CULTIST_0"}
        ).state
        self.assertEqual(state.enemies[0].hp, 20)
        self.assertEqual(strike.enemies[0].hp, 14)
        neutralize = self.simulator.step(
            strike, {"action": "play_card", "card_index": 0, "target": "CULTIST_0"}
        ).state
        self.assertEqual(neutralize.enemies[0].hp, 10)
        self.assertEqual(neutralize.enemies[0].powers[0].power_id, "WEAK")
        self.assertEqual(neutralize.enemies[0].powers[0].amount, 2)
        defended = self.simulator.step(
            neutralize, {"action": "play_card", "card_index": 0}
        ).state
        self.assertEqual(defended.player.block, 5)
        self.assertEqual(defended.energy, 1)

    def test_survivor_starts_and_resolves_discard_selection(self) -> None:
        state = _state(
            CardInstance(1, "SURVIVOR"),
            CardInstance(2, "STRIKE_SILENT"),
        )
        played = self.simulator.step(
            state, {"action": "play_card", "card_index": 0}
        ).state
        self.assertEqual(played.player.block, 8)
        self.assertEqual(played.pending_selection.kind, "discard_from_hand")
        self.assertEqual(
            self.simulator.legal_actions(played),
            [{"action": "combat_select_card", "card_index": 0}],
        )
        selected = self.simulator.step(
            played, {"action": "combat_select_card", "card_index": 0}
        ).state
        self.assertIsNone(selected.pending_selection)
        self.assertEqual(len(selected.hand), 0)
        self.assertEqual([card.uid for card in selected.discard_pile], [1, 2])

    def test_unsupported_turn_and_card_fail_loudly(self) -> None:
        state = _state(CardInstance(1, "STRIKE_SILENT"))
        with self.assertRaises(UnsupportedMechanic):
            self.simulator.step(state, {"action": "end_turn"})
        unknown = _state(CardInstance(9, "RICOCHET"))
        with self.assertRaises(UnsupportedMechanic):
            self.simulator.step(
                unknown, {"action": "play_card", "card_index": 0}
            )

    def test_ricochet_uses_the_sts2_combat_target_rng(self) -> None:
        state = CombatState(
            **{
                **_state(CardInstance(1, "RICOCHET")).__dict__,
                "combat_targets_rng": StsRngState.combat_targets("R54QW50LET"),
            }
        )
        result = self.simulator.step(
            state, {"action": "play_card", "card_index": 0}
        )
        self.assertEqual(result.state.energy, 1)
        self.assertEqual(sum(20 - enemy.hp for enemy in result.state.enemies), 12)
        self.assertEqual(result.state.combat_targets_rng.counter, 4)

    def test_rng_matches_the_net9_game_oracle(self) -> None:
        rng = StsRngState.combat_targets("R54QW50LET")
        values = []
        for _ in range(8):
            value, rng = rng.next_int(2)
            values.append(value)
        self.assertEqual(values, [1, 0, 0, 1, 1, 0, 1, 1])
        self.assertEqual(rng.counter, 8)

    def test_audits_native_replay_card_coverage(self) -> None:
        payload = {
            "format": "sts2_combat_replay_summary",
            "replay": {"players": [{"character": "SILENT"}]},
            "actions": [
                {"action_type": "NetPlayCardAction", "card_id": "STRIKE_SILENT"},
                {"action_type": "NetPlayCardAction", "card_id": "ACROBATICS"},
                {"action_type": "NetEndPlayerTurnAction"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "replay.json"
            replay.write_text(json.dumps(payload), encoding="utf-8")
            result = audit_replay_coverage(replay, self.catalog)
        self.assertTrue(result["character_match"])
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["missing_card_plays"], {"CARD.ACROBATICS": 1})


if __name__ == "__main__":
    unittest.main()
