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
            let signal = ImageVector(width/2, fromSignal)
            signal.ImageFunc
            |> convolve 3 (gauss 2.0)
            |> expand

        let reconstructSignal (fromSignal:VectorND) = 
            ((reconstruct fromSignal)
            |> imageToSignal width).signal

        let inAsSignal = (imageToSignal width inImage).signal

        genRandomVector (-10.0,10.0) (width*width/4)
        |> optimize (quadraticLoss sparsityPenalty inAsSignal reconstructSignal)
        |> function
            (finalSignal, finalLoss) -> 
                reconstruct finalSignal


