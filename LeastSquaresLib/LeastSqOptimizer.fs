﻿namespace LeastSquaresLib

module LeastSqOptimizer =

    open LeastSquaresLib.Helper
    open LeastSquaresLib.VectorND

    // TODO: define some types
    type SignalType = VectorND
    type OptimizerParameters = list<float>

    type LossFunction = OptimizerParameters->float

    // calculate quadratic loss between
    //      * target as array of float
    //      * evaluation of currentFunc at params
    let quadraticLoss 
                (target:SignalType) 
                (currentFunc:OptimizerParameters->SignalType) 
                (forParams:OptimizerParameters) =
        (target.values, (currentFunc forParams).values) 
            ||> Array.map2 (-) 
            |> Array.sumBy (fun x -> x*x)

    // delta parameter controls numerical approximation of gradient
    let delta = 1e-7

    // gradient of loss function with respect to parameter vector
    let dLoss_dParam
                (lossFunc:LossFunction) 
                (atParams:OptimizerParameters) : OptimizerParameters =
        let loss = lossFunc atParams
        atParams
        |> Seq.mapi
            (fun outer _
                -> atParams
                |> List.mapi 
                    (fun inner el 
                        -> if (inner = outer) 
                            then el+delta 
                            else el))
        |> Seq.map lossFunc
        |> Seq.map ((+) -loss)
        |> Seq.map ((*) (1.0/delta))
        |> List.ofSeq

    // update rate determines how much each update pulls the parameters
    let rate = 0.25

    let percentDifference previous updated = 
        (previous, updated)
        ||> List.map2 
            (fun updatedEl currentEl 
                -> 100.0 * abs(updatedEl - currentEl)
                    /(delta + abs(currentEl)))

    // Seq.unfold-ready function to update parameter vector 
    //      given current loss function values
    let unfoldLossFunc 
                (lossFunc:LossFunction) 
                (currentParams:OptimizerParameters, currentLoss:float) =    
            
        let gradient = 
            currentParams 
            |> dLoss_dParam lossFunc

        let updatedParams = 
            (currentParams, 
                gradient)
            ||> List.map2 (fun currEl gradEl -> currEl - rate * gradEl)

        let updatedLoss = 
            updatedParams 
            |> lossFunc

        System.Diagnostics.Trace.Assert(updatedLoss < currentLoss)

        updatedParams
        |> lossFunc
        |> dump "updated loss"
        |> function 
            updatedLoss ->                            
                if 0.01 < abs(updatedLoss - currentLoss)
                then Some ((updatedParams, updatedLoss), 
                            (updatedParams, updatedLoss))
                else None

    // unfold operation on the loss function, starting from the initial parameters
    let optimize (lossFunc:LossFunction) (initParams:OptimizerParameters) = 
        Seq.unfold (unfoldLossFunc lossFunc) (initParams, lossFunc initParams)
        |> List.ofSeq
        |> List.last