namespace LeastSquaresLib

module ImageOptimization =

    open LeastSquaresLib.ImageFilters
    open LeastSquaresLib.LeastSqOptimizer
    open LeastSquaresLib.VectorND
    open LeastSquaresLib.ImageVector

    let nullSparsityPenalty (_:VectorND) = 0.0

    let matchReconstruct sparsityPenalty width inImage : ImageFunc =
        // calculate quadratic loss between
        //      * target as array of float
        //      * evaluation of currentFunc at params
        let quadraticLoss 
                    (sparsityPenalty:VectorND->float) 
                    (target:VectorND) 
                    (currentFunc:VectorND->VectorND) 
                    (forParams:VectorND) =
            let currentValue = 
                currentFunc forParams
            let quadLoss = normL2 (currentValue - target)
            let quadLossAndSparsity = quadLoss + (sparsityPenalty forParams)
            quadLossAndSparsity

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

