namespace LeastSquaresLib

module ImageIO = 

    open LeastSquaresLib.VectorND
    open LeastSquaresLib.ImageFilters
    open LeastSquaresLib.ImageOptimization
    open LeastSquaresLib.ImageVector
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
            signal -> ImageVector(width, signal)

