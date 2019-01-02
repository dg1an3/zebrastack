namespace LeastSquaresLib

module LeastSqOptimizer =

    open LeastSquaresLib.Helper
    open LeastSquaresLib.VectorND
    open LeastSquaresLib.NumericalGradient

    type LossFunction = VectorND->float

    // calculate quadratic loss between
    //      * target as array of float
    //      * evaluation of currentFunc at params
    let quadraticLoss 
                (sparsityPenalty:VectorND->float) 
                (target:VectorND) 
                (currentFunc:VectorND->VectorND) 
                (forParams:VectorND) =
        let currentValue = 
            currentFunc forParams
        let quadLoss = normL2 (currentValue - target)
        let quadLossAndSparsity = quadLoss + (sparsityPenalty forParams)
        quadLossAndSparsity

    let nullSparsityPenalty (_:VectorND) = 0.0

    // update rate determines how much each update pulls the parameters
    let rate = 0.25
    
    // Seq.unfold-ready function to update parameter vector 
    //      given current loss function values
    let unfoldLossFunc 
                (lossFunc:LossFunction) 
                (currentParams:VectorND, currentLoss:float) =    
            
        let gradientAtCurrent = 
            currentParams 
            |> gradient lossFunc

        let updatedParams = 
            (currentParams.values, 
                gradientAtCurrent.values)
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