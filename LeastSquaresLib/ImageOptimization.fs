namespace LeastSquaresLib

module ImageOptimization =

    open LeastSquaresLib.ImageFilters
    open LeastSquaresLib.LeastSqOptimizer
    open LeastSquaresLib.VectorND
    open LeastSquaresLib.ImageVector

    let matchReconstruct sparsityPenalty width inImage : ImageFunc =
        let reconstruct (fromSignal:VectorND) = 
            
            let signal = ImageVector(width, fromSignal)
            signal.ImageFunc
            |> convolve 2 (gauss 1.0)
            // |> asciiImage 10
            |> ignore
            signal

        let reconstructSignal (fromSignal:VectorND) = 
            (reconstruct fromSignal).signal

        let inAsSignal = (imageToSignal width inImage).signal

        genRandomVector (-1.0,1.0) (width*width)
        |> optimize (quadraticLoss sparsityPenalty inAsSignal reconstructSignal)
        |> function
            (finalSignal, finalLoss) -> 
                (reconstruct finalSignal).ImageFunc

