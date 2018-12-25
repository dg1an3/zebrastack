namespace LeastSquaresLib

module ImageOptimization =

    open LeastSquaresLib.ImageFilters
    open LeastSquaresLib.LeastSqOptimizer
    open LeastSquaresLib.VectorND

    type ImageAsSignal = { width:int; signal:VectorND }

    (* helper to expose a signal as an image *)
    let imageFromSignal { width=width; signal=signal } x y =
        let index = y*width + x
        if 0 < x && x < width
            && 0 < index && index < signal.values.Length
        then signal.[index]
        else 0.0

    let imageToSignal width imageFunc =
        { width = width;
            signal = 
                (seq {0..width-1}, seq {0..width-1})
                ||> Seq.allPairs
                |> Seq.map (fun (x,y) -> imageFunc x y)
                |> Array.ofSeq
                |> VectorND }        

    let gaussBasisReconstruct (imageAsSignal:ImageAsSignal) : ImageFunc =
        imageAsSignal
        |> imageFromSignal
        // |> convolve 2 (gauss 2.0)
        // |> expand

    let matchReconstruct width inImage : ImageFunc =
        let reconstruct (fromSignal:OptimizerParameters) = 
            let signal = 
                { signal = fromSignal |> Array.ofList |> VectorND;
                    width = width }
            signal 
            |> imageFromSignal
            // |> asciiImage 10
            |> ignore
            signal

        let reconstructSignal (fromSignal:OptimizerParameters) = 
            (reconstruct fromSignal).signal

        let { signal=inAsSignal } = imageToSignal width inImage

        (genRandomVector (0.0,1.0) (width*width)).values
        |> List.ofArray
        |> optimize (quadraticLoss inAsSignal reconstructSignal)
        |> function
            (finalSignal, finalLoss) -> 
                reconstruct finalSignal
        |> imageFromSignal
