using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Topshelf;

namespace DirCheck
{
    class Check
    {
        private FileSystemWatcher watcher;
        private string sourceFolder;
        private string destinationFolder;

        // custom constructor
        public Check(string sourceFolder, string destinationFolder)
        {
            this.sourceFolder = sourceFolder;
            this.destinationFolder = destinationFolder;
            
            watcher = new FileSystemWatcher(sourceFolder);
            watcher.NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastAccess | NotifyFilters.LastWrite;
        }

        public void Start()
        {
            watcher.Created += OnChanged;
            watcher.Changed += OnChanged;
            watcher.Renamed += OnChanged;
            
            watcher.EnableRaisingEvents = true;
            watcher.IncludeSubdirectories = true;

            Console.WriteLine("The service Check Directory started");
        }

        private void OnChanged(object sender, FileSystemEventArgs e)
        {
            Console.WriteLine("OnChangeEvent has is called");

            string[] files = Directory.GetFiles(sourceFolder, "*.*", SearchOption.AllDirectories);
            int count = files.Length;

            if (count > 100)
            {
                foreach (string file in Directory.GetFiles(sourceFolder, "*.*", SearchOption.AllDirectories))
                {
                    FileInfo fi = new FileInfo(file);

                    if (fi.LastAccessTime < DateTime.Now.AddMinutes(-60))
                    {
                        Console.WriteLine(file + "  should move to:  " + Path.Combine(destinationFolder, Path.GetFileName(file)));
                        File.Move(file, Path.Combine(destinationFolder, Path.GetFileName(file)));
                    }
                }
            }
        }

        public void Stop()
        {
            Console.WriteLine("The Service Check Directory stopped");
            watcher.EnableRaisingEvents = false;
        }
    }
}