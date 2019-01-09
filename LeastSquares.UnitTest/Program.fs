module Program

open LeastSquares.UnitTest
open LeastSquaresLib.AsciiGraph
open LeastSquaresLib.LeastSqOptimizer
open LeastSquaresLib.ImageFilters
open LeastSquaresLib.ImageOptimization
open LeastSquaresLib.ImageIO



let [<EntryPoint>] main _ = 

    let fx = (fun (x:float[]) -> x.[0] + 1.0)
    let gx = (fun (x:float[]) -> x.[0] * 1.0)

    printfn "%f" (fx [| 2.0 |])

    let comp = fx >> (+)
    printfn "%f" (comp [| 3.0 |] -1.0)

    circle 5
    |> shift -5 -5
    |> function 
        image -> 
            image
            |> asciiImage (seq{0..10}) 
            |> Seq.iter (printfn "%s")
            image    
            |> matchReconstruct 16
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
