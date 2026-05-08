using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace HorseLab.Tools
{
    internal static class JvLinkRtDump
    {
        private const int DefaultBufferSize = 256000;

        [STAThread]
        private static int Main(string[] args)
        {
            var options = ParseArgs(args);
            var dataSpec = GetRequired(options, "--data-spec");
            var key = GetRequired(options, "--key");
            var reader = GetString(options, "--reader", "jvgets").ToLowerInvariant();
            var softwareId = GetString(options, "--software-id", "UNKNOWN");
            var outputPath = GetString(options, "--output", @"C:\horse-lab\data\raw\jravan\jvrtdata.txt");
            var logPath = GetString(options, "--log", @"C:\horse-lab\data\raw\jravan\jvlink_rt_dump.log");
            var maxReadIterations = GetInt(options, "--max-read-iterations", 1000000);
            var bufferSize = GetInt(options, "--buffer-size", DefaultBufferSize);

            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
            Directory.CreateDirectory(Path.GetDirectoryName(logPath));
            File.WriteAllText(logPath, string.Empty, Encoding.UTF8);

            object jv = null;
            try
            {
                Log(logPath, string.Format("starting DataSpec={0} Key={1} Reader={2}", dataSpec, key, reader));
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

                int openCode = link.JVRTOpen(dataSpec, key);
                Log(logPath, string.Format("JVRTOpen returned {0}", openCode));
                if (openCode != 0)
                {
                    throw new InvalidOperationException(string.Format("JVRTOpen failed with return code {0}", openCode));
                }

                var shiftJis = Encoding.GetEncoding(932);
                var recordChunks = 0;
                var fileMarkers = 0;
                var reachedEnd = false;

                using (var writer = new StreamWriter(outputPath, false, shiftJis))
                {
                    for (var i = 0; i < maxReadIterations; i++)
                    {
                        int readCode;
                        string recordText = null;
                        string bufferName;

                        if (reader == "jvread")
                        {
                            string buffer = new string('\0', bufferSize);
                            bufferName = string.Empty;
                            readCode = link.JVRead(ref buffer, bufferSize, ref bufferName);
                            if (readCode > 0)
                            {
                                recordText = CleanRecordText(buffer, readCode);
                            }
                        }
                        else if (reader == "jvgets")
                        {
                            var buffer = new byte[bufferSize];
                            bufferName = string.Empty;
                            readCode = link.JVGets(ref buffer, bufferSize, ref bufferName);
                            if (readCode > 0)
                            {
                                recordText = shiftJis.GetString(buffer, 0, readCode).TrimEnd('\0', '\r', '\n');
                            }
                        }
                        else
                        {
                            throw new ArgumentException("Unsupported reader: " + reader);
                        }

                        if (readCode > 0)
                        {
                            writer.WriteLine(recordText);
                            recordChunks += 1;
                            if (recordChunks % 100 == 0)
                            {
                                Log(logPath, string.Format("{0} chunks={1}", reader, recordChunks));
                            }
                            continue;
                        }

                        if (readCode == -1)
                        {
                            fileMarkers += 1;
                            Log(logPath, string.Format("{0} file marker {1} name={2}", reader, fileMarkers, bufferName));
                            continue;
                        }

                        if (readCode == 0)
                        {
                            reachedEnd = true;
                            Log(logPath, string.Format("{0} EOF chunks={1} fileMarkers={2}", reader, recordChunks, fileMarkers));
                            break;
                        }

                        throw new InvalidOperationException(string.Format("{0} failed with return code {1}", reader, readCode));
                    }
                }

                if (!reachedEnd)
                {
                    Log(logPath, string.Format(
                        "{0} max iterations reached chunks={1} fileMarkers={2}",
                        reader,
                        recordChunks,
                        fileMarkers));
                }

                Console.WriteLine(
                    "{{\"outputPath\":\"{0}\",\"dataSpec\":\"{1}\",\"key\":\"{2}\",\"reader\":\"{3}\",\"recordChunks\":{4},\"fileMarkers\":{5}}}",
                    EscapeJson(outputPath),
                    EscapeJson(dataSpec),
                    EscapeJson(key),
                    EscapeJson(reader),
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

        private static string CleanRecordText(string value, int readCode)
        {
            var length = Math.Min(readCode, value.Length);
            var text = value.Substring(0, length);
            var nullIndex = text.IndexOf('\0');
            if (nullIndex >= 0)
            {
                text = text.Substring(0, nullIndex);
            }

            return text.TrimEnd('\0', '\r', '\n');
        }

        private static string EscapeJson(string value)
        {
            return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
        }
    }
}
