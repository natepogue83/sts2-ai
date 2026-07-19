using Mono.Cecil;
using Mono.Cecil.Cil;

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: Sts2CliPatcher <copied-sts2.dll>");
    return 2;
}

string dllPath = Path.GetFullPath(args[0]);
if (!File.Exists(dllPath))
{
    Console.Error.WriteLine($"sts2.dll was not found: {dllPath}");
    return 2;
}

var resolver = new DefaultAssemblyResolver();
resolver.AddSearchDirectory(Path.GetDirectoryName(dllPath)!);
var module = ModuleDefinition.ReadModule(
    dllPath,
    new ReaderParameters
    {
        AssemblyResolver = resolver,
        ReadingMode = ReadingMode.Deferred,
    });

int patches = 0;
foreach (TypeDefinition type in module.Types)
{
    foreach (MethodDefinition method in type.Methods)
    {
        if (method.Name != "WaitUntilQueueIsEmptyOrWaitingOnNonPlayerDrivenAction"
            || method.Body == null)
        {
            continue;
        }

        ILProcessor il = method.Body.GetILProcessor();
        il.Body.Instructions.Clear();
        MethodReference completedTask = module.ImportReference(
            typeof(Task).GetProperty(nameof(Task.CompletedTask))!.GetMethod!);
        il.Emit(OpCodes.Call, completedTask);
        il.Emit(OpCodes.Ret);
        patches++;
    }
}

if (patches != 1)
{
    Console.Error.WriteLine(
        $"Expected one queue-wait method in this game build; found {patches}. Refusing output.");
    return 3;
}

string temporary = dllPath + ".patched";
File.Delete(temporary);
module.Write(temporary);
module.Dispose();
File.Move(temporary, dllPath, overwrite: true);
Console.WriteLine($"Patched isolated sts2.dll: {dllPath}");
return 0;
