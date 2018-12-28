module Program

open LeastSquares.UnitTest
open LeastSquaresLib.AsciiGraph
open LeastSquaresLib.LeastSqOptimizer
open LeastSquaresLib.ImageFilters
open LeastSquaresLib.ImageOptimization
open LeastSquaresLib.ImageIO



let [<EntryPoint>] main _ = 

    circle 5
    |> shift -5 -5
    |> function 
        image -> 
            image
            |> asciiImage (seq{0..10}) 
            |> Seq.iter (printfn "%s")
            image    
            |> matchReconstruct nullSparsityPenalty 16
    |> function 
        image -> 
            image
            |> asciiImage (seq{0..16}) 
            |> Seq.iter (printfn "%s")
            image
    |> ignore

    //let test = TestLeastSqOptimizer()
    //test.TestSlopeInterceptOptimization() |> ignore
    0
