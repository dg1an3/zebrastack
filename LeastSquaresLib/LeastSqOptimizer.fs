namespace LeastSquaresLib

module LeastSqOptimizer =

    // calculate quadratic loss between
    //      * target as array of float
    //      * evaluation of currentFunc at params
    let quadraticLoss 
                (target:float[]) 
                (currentFunc:list<float>->float[]) 
                (forParams:list<float>) =
        (target, currentFunc forParams) 
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
    let dParam_dLoss 
                (lossFunc:list<float>->float) 
                (atParams:list<float>) : list<float> =
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
        |> Seq.map ((/) delta)
        |> List.ofSeq

    // update rate determines how much each update pulls the parameters
    let rate = 0.95

    // Seq.unfold-ready function to update parameter vector 
    //      given current loss function values
    let unfoldLossFunc 
                (lossFunc:list<float>->float) 
                (currentParams:list<float>, currentLoss:float) = 
        let dParams = dParam_dLoss lossFunc currentParams
        let updatedParams = 
            (currentParams, 
                dParams |> List.map ((*) rate)) 
                ||> List.map2 (-)
        let updatedLoss = lossFunc updatedParams
#if PRINT_UNFOLD_UPDATES
        printfn "updated params = %A (%A %%), loss = %f" 
            updatedParams 
            ((updatedParams, currentParams)
                ||> List.map2 
                    (fun updatedEl currentEl 
                        -> 100.0 * abs(updatedEl - currentEl)/(delta + abs(currentEl))))
            updatedLoss 
#endif
        if abs(updatedLoss - currentLoss) < 0.5
        then None
        else Some ((updatedParams, updatedLoss), (updatedParams, updatedLoss))

    // unfold operation on the loss function, starting from the initial parameters
    let optimize (lossFunc:list<float>->float) (initParams:list<float>) = 
        Seq.unfold (unfoldLossFunc lossFunc) (initParams, lossFunc initParams)
        |> List.ofSeq
        |> List.last
