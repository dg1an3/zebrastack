namespace LeastSquaresLib

module ImageOptimization =

    open LeastSquaresLib.ImageFilters
    open LeastSquaresLib.LeastSqOptimizer
    open LeastSquaresLib.VectorND

    let matchReconstruct useSparsity width inImage : ImageFunc =
        let reconstruct (fromSignal:OptimizerParameters) = 
            
            let signal = 
                { signal = fromSignal;
                    width = width }
            signal.ImageFunc
            |> convolve 2 (gauss 1.0)
            // |> asciiImage 10
            |> ignore
            signal

        let reconstructSignal (fromSignal:OptimizerParameters) = 
            (reconstruct fromSignal).signal

        let inAsSignal = (imageToSignal width inImage).signal

        genRandomVector (-1.0,1.0) (width*width)
        |> optimize (quadraticLoss useSparsity inAsSignal reconstructSignal)
        |> function
            (finalSignal, finalLoss) -> 
                (reconstruct finalSignal).ImageFunc

