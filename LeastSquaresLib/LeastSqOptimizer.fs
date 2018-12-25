namespace LeastSquaresLib

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

    // ensures value x does not get too small (absolute value),
    //      while maintaining its sign
    let stabilize x =
        if x < 0.0
        then x - 0.1
        else x + 0.1

    // delta parameter controls numerical approximation of gradient
    let delta = 0.01

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
        |> Seq.map stabilize
        |> Seq.map ((*) (1.0/delta))
        |> List.ofSeq

    // update rate determines how much each update pulls the parameters
    let rate = 0.99

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
        (currentParams, 
            currentParams
            |> dLoss_dParam lossFunc 
            |> List.map (fun g -> rate / g))
            ||> List.map2 (-)                
            |> dump "updated params"
            |> function
                updatedParams -> 
                    updatedParams
                        |> (percentDifference currentParams) 
                        |> (dump "difference %") |> ignore
                    updatedParams
                    |> lossFunc
                    |> dump "updated loss"
                    |> function 
                        updatedLoss ->                            
                            if 0.1 < abs(updatedLoss - currentLoss)
                            then Some ((updatedParams, updatedLoss), 
                                        (updatedParams, updatedLoss))
                            else None

    // unfold operation on the loss function, starting from the initial parameters
    let optimize (lossFunc:LossFunction) (initParams:OptimizerParameters) = 
        Seq.unfold (unfoldLossFunc lossFunc) (initParams, lossFunc initParams)
        |> List.ofSeq
        |> List.last