module Program

open LeastSquares.UnitTest
open LeastSquaresLib.ImageFilters

let [<EntryPoint>] main _ = 
    
    @"..\..\..\..\..\teapot.jpg"
    |> loadImageAsSignal
    |> imageFromSignal
    |> ((convolve 2 (gauss 0.5)) >> decimate)
    |> ((convolve 2 (gauss 0.5)) >> decimate)
    |> asciiImage 30
    |> function
        downsampled -> 
            matchReconstruct (200/4) downsampled
    |> asciiImage 60
    |> ignore

    //let test = TestLeastSqOptimizer()
    //test.TestSlopeInterceptOptimization() |> ignore
    0
