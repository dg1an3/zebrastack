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
        {xmin=0; xmax=loaded.Width; ymin=0; ymax=loaded.Height}
        |> function 
            bounds -> 
                bounds.allIndices
                |> Seq.map (fun (x,y) -> (float loaded.[y,x].B) / 255.0)
                |> (Array.ofSeq >> VectorND)
                |> function signal -> ImageVector(loaded.Width, bounds, signal)

