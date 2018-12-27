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
            target
            |> dump "current params"

        let loss = 
            quadraticLoss false target id currentParams
            |> dump "loss"
        
        Assert.IsTrue(abs(loss) < 1e-8)

    [<TestMethod>]
    member __.TestdLoss_dParam() = 
        let target = 
            [| 0.0; 3.0; 5.0; -2.0; |] 
            |> VectorND
            |> dump "target"

        let approxGradOfShift shift = 
            target.values 
            |> Array.map ((+) shift)
            |> VectorND
            |> dump "params"
            |> dLoss_dParam (quadraticLoss false target id)
            |> dump "grad"
            
        let exactGradOfShift shift =
            target.values
            |> Array.map (fun _ -> 2.0 * shift)
            |> VectorND

        [ 0.0; -5.0; 10.0 ]
        |> List.map 
            (fun shift ->
                ((exactGradOfShift shift).values, 
                    (approxGradOfShift shift).values)
                    ||> Array.forall2 (fun l r -> abs(l-r) < 1e-6))
        |> List.forall id
        |> Assert.IsTrue
                    
        ((approxGradOfShift 0.0).values, 
            (approxGradOfShift 5.0).values) 
            ||> Array.forall2 (fun g sg -> abs(g) < abs(sg))
            |> Assert.IsTrue

        ((approxGradOfShift -5.0).values, 
            (approxGradOfShift -10.0).values) 
            ||> Array.forall2 (fun g sg -> abs(g) < abs(sg))
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

        iter0
        |> optimize (quadraticLoss false target id)
        |> function
            (finalParams, finalLoss) ->                
                printfn "final: params = %A; value = %A; loss = %f" 
                        finalParams
                        finalParams
                        finalLoss
                let initLoss = 
                    quadraticLoss false target id iter0
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

        [| initSlope; initOffset |]
        |> VectorND
        |> optimize (quadraticLoss false target (currentFromSlopeOffset iter0))
        |> function
            (finalParams, finalLoss) ->    
                let [|finalSlope; finalOffset|] = finalParams.values
                printfn "final: slope = %f; offset = %f; value = %A; loss = %f" 
                        finalSlope finalOffset
                        (currentFromSlopeOffset iter0 finalParams)
                        finalLoss
                let initLoss = 
                    quadraticLoss false target (currentFromSlopeOffset iter0) 
                        ([| initSlope; initOffset |] |> VectorND)
                finalLoss < initLoss
        |> Assert.IsTrue
