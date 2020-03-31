using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FileMover;
using Topshelf;

namespace DirCheck
{
    class Program
    {
        //public static string sourceFolder = @"C:\Users\Richard\Desktop\source";
        // fore tests: public static string sourceFolder = @"C:\Users\Hermeler\Desktop\source";
        public static string sourceFolder = @"C:\ATS\Images";
        // public static string destinationFolder = @"C:\Users\Richard\Desktop\destination";
        // public static string destinationFolder = @"C:\Users\Hermeler\Desktop\destination";
        public static string destinationFolder = @"E:\ATS\Images";



        static void Main(string[] args)
        {
            // FileGenerator fileGenerator = new FileGenerator();
            // fileGenerator.Run();

            var exitCode = HostFactory.Run(x =>
            {
                x.Service<Check>(s =>
                {
                    s.ConstructUsing(check => new Check(sourceFolder, destinationFolder));
                    s.WhenStarted(check => check.Start());
                    s.WhenStopped(check => check.Stop());
                });

                x.StartAutomatically(); // Start the service automatically

                // TODO:
                // x.AddCommandLineDefinition("path", v => path = v);
                // x.AddCommandLineDefinition("fileAmount", v => fileAmount = Int32.Parse(v));

                x.RunAsLocalSystem();
                x.SetServiceName("DirCheck");
                x.SetDisplayName("Directory Check");
                x.SetDescription("Check directory and move files \nMAKE SURE 'E:/ATS/Images' IS AVAILABLE");
            });


            int exitCodeValue = (int)Convert.ChangeType(exitCode, exitCode.GetTypeCode());
            Environment.ExitCode = exitCodeValue;
        }
    }
}
