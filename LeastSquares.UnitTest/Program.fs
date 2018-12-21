module Program

open LeastSquares.UnitTest
open LeastSquaresLib.ImageFilters

let [<EntryPoint>] main _ = 

    //"..\..\..\..\..\skull.jpg"
    //|> loadImageAsSignal 
    //|> imageFromSignal 
    //|> ((convolve 2 (gauss 0.5)) >> decimate)
    //|> ((convolve 2 (gauss 0.5)) >> decimate)
    //|> ((convolve 2 (gauss 0.5)) >> decimate)
    //|> asciiImage 60
    
    let test = TestLeastSqOptimizer()
    test.TestSlopeInterceptOptimization() |> ignore
    0
