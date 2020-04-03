using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Timers;
using DirCheck;
using Timer = System.Timers.Timer;

namespace FileMover
{

    /// <summary>
    /// 
    /// </summary>
    class FileGenerator
    {
        public void Run()
        {
            var timer = new System.Threading.Timer(
                e => MyMethod(),
                null,
                TimeSpan.Zero,
                TimeSpan.FromMinutes(0.1));
        }
        
        static void MyMethod()
        {
            int width = 40, height = 20;

            //bitmap
            Bitmap bmp = new Bitmap(width, height);

            //random number
            Random rand = new Random();

            //create random pixels
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    //generate random ARGB value
                    int a = rand.Next(256);
                    int r = rand.Next(256);
                    int g = rand.Next(256);
                    int b = rand.Next(256);

                    //set ARGB value
                    bmp.SetPixel(x, y, Color.FromArgb(a, r, g, b));
                }
            }
            
            //save (write) random pixel image
            bmp.Save(Program.sourceFolder + "\\RandomImage"+ DateTime.Now.Millisecond + ".bmp");
            Console.WriteLine("image saved! at " + DateTime.Now);
        }
    }
}
