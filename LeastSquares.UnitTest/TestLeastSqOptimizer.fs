namespace LeastSquares.UnitTest

open System
open Microsoft.VisualStudio.TestTools.UnitTesting

open LeastSquaresLib.Helper
open LeastSquaresLib.LeastSqOptimizer
open LeastSquaresLib.SlopeInterceptObjective

[<TestClass>]
type TestLeastSqOptimizer() =

    [<TestMethod>]
    member __.TestQuadraticLoss() = 

        // random target
        // TODO: use VectorND for this
        // TODO: use FsCheck to generate this at different lengths etc.
        let target = [| 0.0; 3.0; 5.0; -2.0; |]        
        printfn "target = %A" target

        // generate parameters = target
        let currentParams = target |> List.ofArray
        printfn "currentParams = %A" currentParams

        let currentFunc forParams = forParams |> Array.ofList
        let loss = quadraticLoss target currentFunc currentParams
        printfn "loss = %f" loss

        Assert.IsTrue(abs(loss) < 1e-8)

    [<TestMethod>]
    member __.TestStabilize() =
        let randomArray = genRandomNumbers (-0.1, 0.1) 100
        let checkStabilizeSign = 
            randomArray
            |> Array.forall (fun x -> sign(x) = sign(stabilize x))    
        Assert.IsTrue(checkStabilizeSign)

    [<TestMethod>]
    member __.Test_dParam_dLoss() = 
        let target = [| 0.0; 3.0; 5.0; -2.0; |]        
        printfn "target = %A" target

        // generate parameters = target
        let currentParams = target |> List.ofArray
        printfn "currentParams = %A" currentParams

        let currentFunc forParams = forParams |> Array.ofList
        let grad = dParam_dLoss (quadraticLoss target currentFunc) currentParams
        printfn "little grad = %A" grad
        Assert.IsTrue(grad |> List.forall (fun x -> abs(x) < 0.1))

        let currentParams = target |> List.ofArray |> List.map ((+) 10.0)
        printfn "currentParams = %A" currentParams

        let grad = dParam_dLoss (quadraticLoss target currentFunc) currentParams
        printfn "big grad = %A" grad
        Assert.IsTrue(grad |> List.forall (fun x -> abs(x) < 0.1))

    [<TestMethod>]
    member __.TestSlopeInterceptOptimization() =      

        let target = [| 0.0; 3.0; 5.0; -2.0; |]
        printfn "target = %A" target

        let iter0 = genRandomNumbers (0.0, 10.0) 4
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
