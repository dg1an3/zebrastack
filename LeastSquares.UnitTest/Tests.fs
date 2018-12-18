namespace LeastSquares.UnitTest

open System
open Microsoft.VisualStudio.TestTools.UnitTesting
open LeastSquaresLib

[<TestClass>]
type TestClass () =

    [<TestMethod>]
    member this.TestMethodPassing () =      
        let target = [| 0.0; 3.0; 5.0; -2.0; |]
        let iter0 = Helper.genRandomNumbers 4
        Assert.IsTrue(true);
