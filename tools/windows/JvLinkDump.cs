using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace HorseLab.Tools
{
    internal static class JvLinkDump
    {
        private const int DefaultBufferSize = 110000;

        [STAThread]
        private static int Main(string[] args)
        {
            var options = ParseArgs(args);
            var dataSpec = GetRequired(options, "--data-spec");
            var fromDate = GetRequired(options, "--from-date");
            var option = GetInt(options, "--option", 1);
            var softwareId = GetString(options, "--software-id", "UNKNOWN");
            var outputPath = GetString(options, "--output", @"C:\horse-lab\data\raw\jravan\jvdata.txt");
            var logPath = GetString(options, "--log", @"C:\horse-lab\data\raw\jravan\jvlink_dump.log");
            var maxReadIterations = GetInt(options, "--max-read-iterations", 1000000);

            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
            Directory.CreateDirectory(Path.GetDirectoryName(logPath));
            File.WriteAllText(logPath, string.Empty, Encoding.UTF8);

            object jv = null;
            try
            {
                Log(logPath, string.Format("starting DataSpec={0} FromDate={1} Option={2}", dataSpec, fromDate, option));
                var jvType = Type.GetTypeFromProgID("JVDTLab.JVLink", true);
                jv = Activator.CreateInstance(jvType);
                dynamic link = jv;
                Log(logPath, "created COM object");

                int initCode = link.JVInit(softwareId);
                Log(logPath, string.Format("JVInit returned {0}", initCode));
                if (initCode != 0)
                {
                    throw new InvalidOperationException(string.Format("JVInit failed with return code {0}", initCode));
                }

                int readCount = 0;
                int downloadCount = 0;
                string lastFileTimestamp = string.Empty;
                int openCode = link.JVOpen(
                    dataSpec,
                    fromDate,
                    option,
                    ref readCount,
                    ref downloadCount,
                    ref lastFileTimestamp);
                Log(logPath, string.Format(
                    "JVOpen returned {0} readCount={1} downloadCount={2} lastFileTimestamp={3}",
                    openCode,
                    readCount,
                    downloadCount,
                    lastFileTimestamp));
                if (openCode != 0)
                {
                    throw new InvalidOperationException(string.Format("JVOpen failed with return code {0}", openCode));
                }

                var shiftJis = Encoding.GetEncoding(932);
                var recordChunks = 0;
                var fileMarkers = 0;
                var reachedEnd = false;

                using (var writer = new StreamWriter(outputPath, false, shiftJis))
                {
                    for (var i = 0; i < maxReadIterations; i++)
                    {
                        var buffer = new byte[DefaultBufferSize];
                        string bufferName = string.Empty;
                        int readCode = link.JVGets(ref buffer, DefaultBufferSize, ref bufferName);

                        if (readCode > 0)
                        {
                            writer.WriteLine(shiftJis.GetString(buffer, 0, readCode).TrimEnd('\r', '\n'));
                            recordChunks += 1;
                            if (recordChunks % 100 == 0)
                            {
                                Log(logPath, string.Format("JVGets chunks={0}", recordChunks));
                            }
                            continue;
                        }

                        if (readCode == -1)
                        {
                            fileMarkers += 1;
                            Log(logPath, string.Format("JVGets file marker {0} name={1}", fileMarkers, bufferName));
                            continue;
                        }

                        if (readCode == 0)
                        {
                            reachedEnd = true;
                            Log(logPath, string.Format("JVGets EOF chunks={0} fileMarkers={1}", recordChunks, fileMarkers));
                            break;
                        }

                        throw new InvalidOperationException(string.Format("JVGets failed with return code {0}", readCode));
                    }
                }

                if (!reachedEnd)
                {
                    Log(logPath, string.Format(
                        "JVGets max iterations reached chunks={0} fileMarkers={1}",
                        recordChunks,
                        fileMarkers));
                }

                Console.WriteLine(
                    "{{\"outputPath\":\"{0}\",\"dataSpec\":\"{1}\",\"fromDate\":\"{2}\",\"option\":{3},\"readCount\":{4},\"downloadCount\":{5},\"lastFileTimestamp\":\"{6}\",\"recordChunks\":{7},\"fileMarkers\":{8}}}",
                    EscapeJson(outputPath),
                    EscapeJson(dataSpec),
                    EscapeJson(fromDate),
                    option,
                    readCount,
                    downloadCount,
                    EscapeJson(lastFileTimestamp),
                    recordChunks,
                    fileMarkers);
                return 0;
            }
            catch (Exception ex)
            {
                Log(logPath, "ERROR " + ex);
                Console.Error.WriteLine(ex);
                return 1;
            }
            finally
            {
                if (jv != null)
                {
                    try
                    {
                        dynamic link = jv;
                        link.JVClose();
                        Log(logPath, "JVClose completed");
                    }
                    catch
                    {
                    }

                    Marshal.FinalReleaseComObject(jv);
                }
            }
        }

        private static Dictionary<string, string> ParseArgs(string[] args)
        {
            var options = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (var i = 0; i < args.Length; i++)
            {
                var key = args[i];
                if (!key.StartsWith("--", StringComparison.Ordinal))
                {
                    throw new ArgumentException("Unexpected argument: " + key);
                }

                if (i + 1 >= args.Length)
                {
                    throw new ArgumentException("Missing value for argument: " + key);
                }

                options[key] = args[++i];
            }

            return options;
        }

        private static string GetRequired(Dictionary<string, string> options, string key)
        {
            string value;
            if (!options.TryGetValue(key, out value) || string.IsNullOrWhiteSpace(value))
            {
                throw new ArgumentException("Missing required argument: " + key);
            }

            return value;
        }

        private static string GetString(Dictionary<string, string> options, string key, string defaultValue)
        {
            string value;
            return options.TryGetValue(key, out value) ? value : defaultValue;
        }

        private static int GetInt(Dictionary<string, string> options, string key, int defaultValue)
        {
            string value;
            return options.TryGetValue(key, out value) ? int.Parse(value) : defaultValue;
        }

        private static void Log(string logPath, string message)
        {
            File.AppendAllText(logPath, DateTime.UtcNow.ToString("o") + " " + message + Environment.NewLine, Encoding.UTF8);
        }

        private static string EscapeJson(string value)
        {
            return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
        }
    }
}
