namespace LeastSquares.UnitTest

open System
open Microsoft.VisualStudio.TestTools.UnitTesting

open LeastSquaresLib.Helper
open LeastSquaresLib.VectorND
open LeastSquaresLib.NumericalGradient
open LeastSquaresLib.LeastSqOptimizer
open LeastSquaresLib.SlopeInterceptObjective
open SixLabors.ImageSharp.Processing

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

        let target = [| 0.0; 3.0; 5.0; -2.0; |] |> VectorND     |> dump "target"

        let iter0 = genRandomVector (-5.0, 5.0) 4               |> dump "iter0"

        iter0
        |> optimize ((-) target >> normL2)
        |> function
            (finalParams, finalLoss) ->                
                printfn "final: params = %A; value = %A; loss = %f" 
                        finalParams
                        finalParams
                        finalLoss
                let initLoss = ((-) target >> normL2) iter0
                finalLoss < initLoss
        |> Assert.IsTrue

    [<Ignore>]
    [<TestMethod>]
    member __.TestSlopeInterceptOptimization() =      

        let target = [| 0.0; 3.0; 5.0; -2.0; |] |> VectorND     |> dump "target"

        let iter0 = genRandomVector (-5.0, 5.0) 4               |> dump "iter0"

        let initSlope = 1.0
        let initOffset = 0.0

        // calculate quadratic loss between
        //      * target as array of float
        //      * evaluation of currentFunc at params
        let quadraticLoss 
                    (target:VectorND) 
                    (currentFunc:VectorND->VectorND) 
                    (forParams:VectorND) =
            let currentValue = 
                currentFunc forParams
            normL2 (currentValue - target)

        [| initSlope; initOffset |]
        |> VectorND
        |> optimize (quadraticLoss target (currentFromSlopeOffset iter0))
        |> function
            (finalParams, finalLoss) ->    
                let [|finalSlope; finalOffset|] = finalParams.values
                printfn "final: slope = %f; offset = %f; value = %A; loss = %f" 
                        finalSlope finalOffset
                        (currentFromSlopeOffset iter0 finalParams)
                        finalLoss
                let initLoss = 
                    quadraticLoss target (currentFromSlopeOffset iter0) 
                        ([| initSlope; initOffset |] |> VectorND)
                finalLoss < initLoss
        |> Assert.IsTrue
