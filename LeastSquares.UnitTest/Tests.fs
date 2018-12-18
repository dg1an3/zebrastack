namespace LeastSquares.UnitTest

open System
open Microsoft.VisualStudio.TestTools.UnitTesting

open LeastSquaresLib.Helper
open LeastSquaresLib.LeastSqOptimizer
open LeastSquaresLib.SlopeInterceptObjective

[<TestClass>]
type TestClass () =

    [<TestMethod>]
    member this.TestSlopeInterceptOptimization() =      

        let target = [| 0.0; 3.0; 5.0; -2.0; |]
        printfn "target = %A" target

        let iter0 = genRandomNumbers 4
        printfn "iter0 = %A" iter0

        let initSlope = 1.0
        let initOffset = 0.0

        optimize 
            (quadraticLoss target (currentFromSlopeOffset iter0)) 
                [initSlope; initOffset]
        |> function
            ([finalSlope; finalOffset], finalLoss) ->
                sprintf "final: slope = %f; offset = %f; value = %A; loss = %f" 
                        finalSlope finalOffset
                        (currentFromSlopeOffset iter0 [finalSlope; finalOffset])
                        finalLoss
        |> function 
            output -> Console.WriteLine(output)

        Assert.IsTrue(true)
