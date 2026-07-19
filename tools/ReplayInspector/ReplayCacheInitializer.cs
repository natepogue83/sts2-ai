using System.Buffers.Binary;
using System.IO.Hashing;
using System.Reflection;
using System.Text;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Timeline;

internal static class ReplayCacheInitializer
{
    private const BindingFlags StaticPrivate = BindingFlags.Static | BindingFlags.NonPublic;

    public static uint InitializeBaseGame()
    {
        Type cacheType = typeof(ModelIdSerializationCache);
        var categoryToId = GetField<Dictionary<string, int>>(
            cacheType, "_categoryNameToNetIdMap");
        var idToCategory = GetField<List<string>>(cacheType, "_netIdToCategoryNameMap");
        var entryToId = GetField<Dictionary<string, int>>(cacheType, "_entryNameToNetIdMap");
        var idToEntry = GetField<List<string>>(cacheType, "_netIdToEntryNameMap");
        var epochToId = GetField<Dictionary<string, int>>(cacheType, "_epochNameToNetIdMap");
        var idToEpoch = GetField<List<string>>(cacheType, "_netIdToEpochNameMap");

        categoryToId.Clear();
        idToCategory.Clear();
        entryToId.Clear();
        idToEntry.Clear();
        epochToId.Clear();
        idToEpoch.Clear();
        categoryToId[ModelId.none.Category] = 0;
        idToCategory.Add(ModelId.none.Category);
        entryToId[ModelId.none.Entry] = 0;
        idToEntry.Add(ModelId.none.Entry);

        byte[] buffer = new byte[512];
        var hash = new XxHash32();
        List<Type> modelTypes = AbstractModelSubtypes.All.OrderBy(type => type.Name).ToList();
        foreach (Type modelType in modelTypes)
        {
            ModelId id = ModelDb.GetId(modelType);
            AddUnique(categoryToId, idToCategory, id.Category);
            AddUnique(entryToId, idToEntry, id.Entry);
            Append(hash, buffer, id.Category);
            Append(hash, buffer, id.Entry);
        }
        foreach (string epochId in EpochModel.AllEpochIds)
        {
            AddUnique(epochToId, idToEpoch, epochId);
            Append(hash, buffer, epochId);
        }

        SetAutoProperty(cacheType, "CategoryIdBitSize", BitsFor(idToCategory.Count));
        SetAutoProperty(cacheType, "EntryIdBitSize", BitsFor(idToEntry.Count));
        SetAutoProperty(cacheType, "EpochIdBitSize", BitsFor(idToEpoch.Count));
        BinaryPrimitives.WriteInt32LittleEndian(buffer, idToCategory.Count);
        hash.Append(buffer.AsSpan(0, 4));
        BinaryPrimitives.WriteInt32LittleEndian(buffer, idToEntry.Count);
        hash.Append(buffer.AsSpan(0, 4));
        BinaryPrimitives.WriteInt32LittleEndian(buffer, idToEpoch.Count);
        hash.Append(buffer.AsSpan(0, 4));
        uint value = hash.GetCurrentHashAsUInt32();
        SetAutoProperty(cacheType, "Hash", value);
        return value;
    }

    private static T GetField<T>(Type type, string name) where T : class
    {
        return (T)(type.GetField(name, StaticPrivate)?.GetValue(null)
            ?? throw new MissingFieldException(type.FullName, name));
    }

    private static void SetAutoProperty(Type type, string name, object value)
    {
        FieldInfo field = type.GetField($"<{name}>k__BackingField", StaticPrivate)
            ?? throw new MissingFieldException(type.FullName, name);
        field.SetValue(null, value);
    }

    private static int BitsFor(int count)
    {
        return (int)Math.Ceiling(Math.Log2(count));
    }

    private static void AddUnique(
        Dictionary<string, int> mapping, List<string> reverse, string value)
    {
        if (mapping.ContainsKey(value))
            return;
        mapping[value] = reverse.Count;
        reverse.Add(value);
    }

    private static void Append(XxHash32 hash, byte[] buffer, string value)
    {
        int bytes = Encoding.UTF8.GetBytes(value, 0, value.Length, buffer, 0);
        hash.Append(buffer.AsSpan(0, bytes));
    }
}
