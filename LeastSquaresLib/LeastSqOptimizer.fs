namespace LeastSquaresLib

module LeastSqOptimizer =

    open LeastSquaresLib.Helper
    open LeastSquaresLib.VectorND

    type LossFunction = VectorND->float

    // calculate quadratic loss between
    //      * target as array of float
    //      * evaluation of currentFunc at params
    let quadraticLoss 
                (sparsityPenalty:VectorND->float) 
                (target:VectorND) 
                (currentFunc:VectorND->VectorND) 
                (forParams:VectorND) =
        currentFunc forParams
        |> (-) target
        |> normL2
        |> (+) (sparsityPenalty forParams)

    let nullSparsityPenalty (_:VectorND) = 0.0

    // delta parameter controls numerical approximation of gradient
    let delta = 1e-7

    // gradient of loss function with respect to parameter vector
    let dLoss_dParam
                (lossFunc:LossFunction) 
                (atParams:VectorND) : VectorND =
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
                (currentParams:VectorND, currentLoss:float) =    
            
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
    let optimize (lossFunc:LossFunction) (initParams:VectorND) = 
        Seq.unfold (unfoldLossFunc lossFunc) (initParams, lossFunc initParams)
        |> List.ofSeq
        |> List.last