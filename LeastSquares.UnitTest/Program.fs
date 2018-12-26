module Program

open LeastSquares.UnitTest
open LeastSquaresLib.ImageFilters
open LeastSquaresLib.ImageOptimization
open LeastSquaresLib.ImageIO


let [<EntryPoint>] main _ = 

    circle 5
    |> shift -5 -5
    |> asciiImage 10
    |> function
        imgCircle -> 
            matchReconstruct 16 imgCircle
    |> asciiImage 16
    |> ignore

    //let test = TestLeastSqOptimizer()
    //test.TestSlopeInterceptOptimization() |> ignore
    0
