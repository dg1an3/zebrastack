namespace LeastSquares.UnitTest

open System
open Microsoft.VisualStudio.TestTools.UnitTesting

open LeastSquaresLib.Helper
open LeastSquaresLib.VectorND
open LeastSquaresLib.LeastSqOptimizer
open LeastSquaresLib.SlopeInterceptObjective

[<TestClass>]
type TestLeastSqOptimizer() =

    [<TestMethod>]
    member __.TestQuadraticLoss() = 

        // random target
        // TODO: use FsCheck to generate this at different lengths etc.
        let target = 
            [| 0.0; 3.0; 5.0; -2.0; |] 
            |> VectorND
            |> dump "target"

        // generate parameters = target
        let currentParams = 
            target.values 
            |> List.ofArray
            |> dump "current params"

        let currentFunc forParams = forParams |> Array.ofList |> VectorND
        let loss = 
            quadraticLoss target currentFunc currentParams
            |> dump "loss"
        
        Assert.IsTrue(abs(loss) < 1e-8)

    [<TestMethod>]
    member __.TestdLoss_dParam() = 
        let target = 
            [| 0.0; 3.0; 5.0; -2.0; |] 
            |> VectorND
            |> dump "target"

        // parameter function that just turns the parameter vector directly in to a VectorND
        let paramsToSignal forParams = 
            forParams 
            |> Array.ofSeq 
            |> VectorND

        let approxGradOfShift shift = 
            target.values 
            |> List.ofArray
            |> List.map ((+) shift)
            |> dump "params"
            |> dLoss_dParam (quadraticLoss target paramsToSignal)
            |> dump "grad"
            
        let exactGradOfShift shift =
            target.values
            |> List.ofArray
            |> List.map (fun _ -> 2.0 * shift)

        (exactGradOfShift 0.0, 
            approxGradOfShift 0.0)
            ||> List.forall2 (fun l r -> abs(l-r) < 1e-6)
            |> Assert.IsTrue

        (exactGradOfShift -5.0, 
            approxGradOfShift -5.0)
            ||> List.forall2 (fun l r -> abs(l-r) < 1e-6)
            |> Assert.IsTrue

        (exactGradOfShift 10.0, 
            approxGradOfShift 10.0)
            ||> List.forall2 (fun l r -> abs(l-r) < 1e-6)
            |> Assert.IsTrue

        (approxGradOfShift 0.0, 
            approxGradOfShift 5.0) 
            ||> List.forall2 (fun g sg -> abs(g) < abs(sg))
            |> Assert.IsTrue

        (approxGradOfShift -5.0, 
            approxGradOfShift -10.0) 
            ||> List.forall2 (fun g sg -> abs(g) < abs(sg))
            |> Assert.IsTrue
            

    [<TestMethod>]
    member __.TestDirectOptimization() =      

        let target = 
            [| 0.0; 3.0; 5.0; -2.0; |] 
            |> VectorND
            |> dump "target"

        let iter0 = 
            genRandomVector (-5.0, 5.0) 4
            |> dump "iter0"

        let directFromParams param = 
            param |> Array.ofList |> VectorND

        iter0.values
        |> List.ofArray
        |> optimize (quadraticLoss target directFromParams)
        |> function
            (finalParams, finalLoss) ->                
                printfn "final: params = %A; value = %A; loss = %f" 
                        finalParams
                        (directFromParams finalParams)
                        finalLoss
                let initLoss = 
                    quadraticLoss target 
                        directFromParams
                        (iter0.values |> List.ofArray)
                finalLoss < initLoss
        |> Assert.IsTrue

    [<TestMethod>]
    member __.TestSlopeInterceptOptimization() =      

        let target = 
            [| 0.0; 3.0; 5.0; -2.0; |] 
            |> VectorND
            |> dump "target"

        let iter0 = 
            genRandomVector (-5.0, 5.0) 4
            |> dump "iter0"

        let initSlope = 1.0
        let initOffset = 0.0

        [initSlope; initOffset]
        |> optimize (quadraticLoss target (currentFromSlopeOffset iter0))
        |> function
            ([finalSlope; finalOffset], finalLoss) ->                
                printfn "final: slope = %f; offset = %f; value = %A; loss = %f" 
                        finalSlope finalOffset
                        (currentFromSlopeOffset iter0 [finalSlope; finalOffset])
                        finalLoss
                let initLoss = 
                    quadraticLoss target 
                        (currentFromSlopeOffset iter0) 
                        [initSlope; initOffset]
                finalLoss < initLoss
        |> Assert.IsTrue
