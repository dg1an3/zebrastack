namespace LeastSquares.UnitTest

open System
open Microsoft.VisualStudio.TestTools.UnitTesting

open LeastSquaresLib.Helper
open LeastSquaresLib.VectorND
open LeastSquaresLib.NumericalGradient
open LeastSquaresLib.LeastSqOptimizer
open LeastSquaresLib.SlopeInterceptObjective
open SixLabors.ImageSharp.Processing

module LeastSqOptimizerTest =

    [<TestClass>]
    type TestLeastSqOptimizer() =

        let verySmall x = abs(x) < 1e-8

        [<TestMethod>]
        member __.TestNumericalGradient() =
        
            let sq (x:VectorND) = x.[0] * x.[0]

            let dSq_dx (x:VectorND) = [| 2.0 * x.[0] |] |> VectorND 

            let compareGradient x =
                let input = [|x|] |> VectorND
                let numerical = gradient sq input               |> dump "numerical"
                let exact = dSq_dx input                        |> dump "exact"
                normL2 (numerical - exact)
                |> verySmall

            [ -8.0; 1.0; 0.8 ]
            |> List.map compareGradient
            |> List.iter Assert.IsTrue

        [<TestMethod>]
        member __.TestQuadraticLoss() = 

            // random target
            let target = [| 0.0; 3.0; 5.0; -2.0; |] |> VectorND     |> dump "target"

            // generate parameters = target
            let currentParams = target                      |> dump "current params"

            currentParams
            |> ((-) target >> normL2)                               |> dump "loss"
            |> verySmall
            |> Assert.IsTrue

        [<TestMethod>]
        member __.TestGradient() = 
            let target = [| 0.0; 3.0; 5.0; -2.0; |] |> VectorND     |> dump "target"

            let approxGradOfShift shift = 
                Array.create 4 shift |> VectorND
                |> ((+) target)                                     |> dump "params"
                |> gradient (((-) target) >> normL2)                |> dump "grad"
            
            let exactGradOfShift shift =
                target.values
                |> Array.map (fun x -> 2.0 * (shift - x))
                |> VectorND

            let approx = approxGradOfShift 0.0
            let exact = exactGradOfShift 0.0

            [ 0.0; -5.0; 10.0 ]
            |> List.map 
                (fun shift ->
                    normL2 ((exactGradOfShift shift) - (approxGradOfShift shift))
                    |> verySmall)
            |> List.iter Assert.IsTrue
                    
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
                [| 0.1; 13.0; 0.6; 24.2; 1.3 |] |> VectorND       |> dump "target"

            let iter0 = genRandomVector (0.0, 50.0) 5             |> dump "iter0"

            let sparsityLambda = 0.2
            let objective x =
                (x |> (-) target |> normL2)
                    + (x |> logSparsity |> (*) sparsityLambda)

            iter0
            |> optimize objective
            |> function
                (finalParams, finalLoss) ->                
                    printfn "target = %A" target
                    printfn "final value = %A" finalParams
                    let initLoss = objective iter0
                    printfn "initLoss %f -> finalLoss %f" (objective iter0) finalLoss                    
                    finalLoss < initLoss
            |> Assert.IsTrue

        // [<Ignore>]
        [<TestMethod>]
        member __.TestSlopeInterceptOptimization() =      

            let target = [| 0.0; 3.0; 5.0; -2.0; |] |> VectorND     |> dump "target"

            let iter0 = genRandomVector (-5.0, 5.0) 4               |> dump "iter0"

            let initSlope = 1.0
            let initOffset = 0.0

            // calculate quadratic loss between
            //      * target as array of float
            //      * evaluation of currentFunc at params
            //let quadraticLoss 
            //            (target:VectorND) 
            //            (currentFunc:VectorND->VectorND) 
            //            (forParams:VectorND) =
            //    let currentValue = 
            //        currentFunc forParams
            //    normL2 (currentValue - target)

            let objective =
                currentFromSlopeOffset iter0
                    >> (-) target
                    >> normL2
            [| initSlope; initOffset |]
            |> VectorND
            // |> optimize (quadraticLoss target (currentFromSlopeOffset iter0))
            |> optimize objective
            |> function
                (finalParams, finalLoss) ->    
                    let [|finalSlope; finalOffset|] = finalParams.values
                    printfn "final: slope = %f; offset = %f; value = %A; loss = %f" 
                            finalSlope finalOffset
                            (currentFromSlopeOffset iter0 finalParams)
                            finalLoss
                    let initLoss = 
                        objective ([| initSlope; initOffset |] |> VectorND)
                    finalLoss < initLoss
            |> Assert.IsTrue
