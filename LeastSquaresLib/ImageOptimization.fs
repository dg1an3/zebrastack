namespace LeastSquaresLib

module ImageOptimization =

    open LeastSquaresLib.ImageFilters
    open LeastSquaresLib.LeastSqOptimizer
    open LeastSquaresLib.VectorND
    open LeastSquaresLib.ImageVector

    let matchReconstruct width inImage : ImageFunc =

        let reconstruct (fromSignal:VectorND) =             
            let signal = ImageVector(width/2, fromSignal)
            imageFuncFromVector signal.Width signal.Vector
            |> convolve 3 (gauss 2.0)
            |> expand

        let reconstructSignal (fromSignal:VectorND) = 
            reconstruct fromSignal
            |> imageVectorFromFunc width

        let inVector = imageVectorFromFunc width inImage

        genRandomVector (-10.0,10.0) (width*width/4)
        |> optimize ((reconstructSignal >> (-) inVector >> normL2) 
                        +>> ((*) 0.01 >> logSparsity))
        |> function
            (finalSignal, finalLoss) -> 
                reconstruct finalSignal


