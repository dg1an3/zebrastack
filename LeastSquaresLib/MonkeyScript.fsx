#r @"bin\Debug\netstandard2.0\LeastSquaresLib.dll"

open System
open System.IO
open System.Drawing
open System.Drawing.Imaging
open System.Diagnostics

let tempFileName ext =
    [ Path.GetTempPath();
        Guid.NewGuid().ToString(); 
        "."; ext ]
    |> String.concat ""

let funcToGraphics (gfx:Graphics) (f:int->int->Color) (x,y) = 
    let color = f x y in 
        use brush = new SolidBrush(color) in 
            gfx.FillRectangle(brush,x,y,1,1)

let funcToBitmap (rng:seq<int>) (f:int->int->Color) =
    let width = (rng |> Seq.max) - (rng |> Seq.min)
    let bitmap = new Bitmap(width,width)
    use graphics = Graphics.FromImage(bitmap)   
    (rng,rng) 
    ||> Seq.allPairs 
    |> Seq.iter (funcToGraphics graphics f)
    bitmap

let saveBitmap (bitmap:Bitmap) =
    ImageFormat.Png.ToString() 
    |> tempFileName 
    |> function 
        fn -> 
            use fs = fn |> File.Create
            bitmap.Save(fs , ImageFormat.Png)
            bitmap.Dispose()
            fn

(fun x y -> let i = 2*x*y|>double|>(*)0.1|>sqrt|>int in Color.FromArgb(i,i,0))
|> funcToBitmap (seq{0..200})
|> saveBitmap
|> ProcessStartInfo
|> Process.Start |> ignore



