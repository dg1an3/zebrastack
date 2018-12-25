namespace LeastSquaresLib

module ImageIO = 

    open LeastSquaresLib.VectorND
    open LeastSquaresLib.ImageFilters
    open LeastSquaresLib.ImageOptimization
    open SixLabors.ImageSharp

    (* loads an image from file, returning the signal (vector) and
        width representations *)
    let loadImageAsSignal (fileName:string) =
        let loaded = Image.Load(fileName)
        let width, height = loaded.Width, loaded.Height
        (seq {0..width-1}, seq {0..height-1})
        ||> Seq.allPairs
        |> Seq.map 
            (fun (x,y) 
                -> (float loaded.[y,x].B) / 255.0)
        |> Array.ofSeq
        |> VectorND
        |> function 
            signal -> { width=width; signal=signal }


    (* ascii image output *)
    let asciiImage range (image:ImageFunc) =
        let range1d = seq {0..range}
        let values = 
            (range1d, range1d) 
            ||> Seq.allPairs
            |> Seq.map (fun (x,y) -> image x y)
        let (minValue, maxValue) = 
            (values |> Seq.min), (values |> Seq.max)
        let asciiPixelArray = [|"   "; " . "; " .."; "..."; "..:"; ".::"; ":::"|]
        let index value = 
            float (asciiPixelArray.Length-1)
                * (value - minValue) / (maxValue - minValue + 1.0)
        let asciiPixel value = 
            asciiPixelArray.[int (index value)]
        range1d
        |> Seq.map (fun row -> 
                System.String.Join("", 
                    range1d |> Seq.map (fun column -> asciiPixel(image column row))))
        |> List.ofSeq
        |> List.iter (printfn "%s")        
        image