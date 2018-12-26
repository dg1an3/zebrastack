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

    let matchReconstruct width inImage : ImageFunc =
        let reconstruct (fromSignal:OptimizerParameters) = 
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
            let signal = 
                { signal = fromSignal;
                    width = width }
            signal 
            |> imageFromSignal
            |> convolve 2 (gauss 1.0)
            // |> asciiImage 10
            |> ignore
            signal

        let reconstructSignal (fromSignal:OptimizerParameters) = 
            (reconstruct fromSignal).signal

        let { signal=inAsSignal } = imageToSignal width inImage

        genRandomVector (-1.0,1.0) (width*width)
        |> optimize (quadraticLoss inAsSignal reconstructSignal)
        |> function
            (finalSignal, finalLoss) -> 
                reconstruct finalSignal
        |> imageFromSignal
