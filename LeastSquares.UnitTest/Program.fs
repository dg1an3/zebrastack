module Program

open LeastSquares.UnitTest
open LeastSquaresLib.ImageFilters

let [<EntryPoint>] main _ = 

    circle 5
    |> asciiImage 10
    |> function
        imgCircle -> 
            matchReconstruct 16 imgCircle
    |> asciiImage 16
    |> ignore

    //let test = TestLeastSqOptimizer()
    //test.TestSlopeInterceptOptimization() |> ignore
    0
