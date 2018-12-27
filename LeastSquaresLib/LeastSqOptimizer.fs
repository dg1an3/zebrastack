namespace LeastSquaresLib

module LeastSqOptimizer =

    open LeastSquaresLib.Helper
    open LeastSquaresLib.VectorND

    // TODO: define some types
    type SignalType = VectorND
    type OptimizerParameters = VectorND

    type LossFunction = OptimizerParameters->float

    // calculate quadratic loss between
    //      * target as array of float
    //      * evaluation of currentFunc at params
    let quadraticLoss 
                (useSparsity:bool)
                (target:SignalType) 
                (currentFunc:OptimizerParameters->SignalType) 
                (forParams:OptimizerParameters) =
        currentFunc forParams
        |> (-) target
        |> normL2
        |> (+) (if useSparsity 
                then normL2 forParams
                else 0.0)

    // delta parameter controls numerical approximation of gradient
    let delta = 1e-7

    // gradient of loss function with respect to parameter vector
    let dLoss_dParam
                (lossFunc:LossFunction) 
                (atParams:OptimizerParameters) : OptimizerParameters =
        let loss = lossFunc atParams
        atParams.values
        |> Array.mapi
            (fun outer _
                -> atParams.values
                |> Array.mapi 
                    (fun inner el 
                        -> if (inner = outer) 
                            then el+delta 
                            else el))
        |> Seq.map VectorND
        |> Seq.map lossFunc
        |> Seq.map ((+) -loss)
        |> Seq.map ((*) (1.0/delta))
        |> Array.ofSeq
        |> VectorND

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
            (currentParams.values, 
                gradient.values)
            ||> Array.map2 (fun currEl gradEl -> currEl - rate * gradEl)
            |> VectorND

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