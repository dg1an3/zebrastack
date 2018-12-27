namespace LeastSquares.UnitTest

open System
open Microsoft.VisualStudio.TestTools.UnitTesting

open LeastSquaresLib.Helper
open LeastSquaresLib.VectorND
open LeastSquaresLib.ImageFilters
open LeastSquaresLib.ImageOptimization
open LeastSquaresLib.ImageIO

[<TestClass>]
type TestImageFilters() =

    let compareImageFuncs range funcLeft funcRight =
        (seq{-range..range}, seq{-range..range})
        ||> Seq.allPairs
        |> Seq.map 
            (fun (x,y) -> abs((funcLeft x y) - (funcRight x y)))
        |> Seq.forall ((>) 1e-4)

    [<TestMethod>]
    member __.TestConvolve() = 

        // dirac image
        let dirac x y =
            match (x, y) with
            | (0, 0) -> 1.0
            | _ -> 0.0

        dirac
        |> convolve 2 (gauss 0.2)
        |> compareImageFuncs 5 (gauss 0.2)
        |> Assert.IsTrue

    [<TestMethod>]
    member __.TestExpandDecimate() =
        circle 5
        |> expand |> asciiImage 20
        |> expand |> asciiImage 20
        |> expand |> asciiImage 20
        |> decimate |> asciiImage 20
        |> decimate |> asciiImage 20
        |> decimate |> asciiImage 20
        |> compareImageFuncs 10 (circle 5)
        |> Assert.IsTrue

    [<TestMethod>]
    member __.TestPyramid() =      

        //"..\..\..\..\..\..\MLData\skull.jpg"
        //|> loadImageAsSignal 
        //|> imageFromSignal
        circle 5
        |> shift -10 -10
        |> ((convolve 2 (gauss 1.0)) >> decimate)
        |> ((convolve 2 (gauss 1.0)) >> decimate)
        |> ((convolve 2 (gauss 1.0)) >> decimate)
        |> asciiImage 60 
        |> function
            level3 -> (level3 0 0) >= 0.0
        |> Assert.IsTrue

    [<TestMethod>]
    member __.TestMatchReconstruct() =
        circle 5
        |> (convolve 2 (gauss 0.5))
        |> shift -5 -5
        |> matchReconstruct false 10
        |> asciiImage 60
        |> function _ -> true
        |> Assert.IsTrue