namespace LeastSquares.UnitTest

open System
open Microsoft.VisualStudio.TestTools.UnitTesting

open LeastSquaresLib.Helper
open LeastSquaresLib.VectorND
open LeastSquaresLib.ImageFilters

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
    member __.TestReconstruction() =

        { width = 3;
            signal = 
                VectorND(
                    [|0.0;0.5;1.0;
                        0.5;1.0;1.5;
                        0.0;0.5;1.0|] )}
        |> gaussBasisReconstruct
        |> asciiImage 20
        |> ignore
        Assert.IsTrue(true)

    [<TestMethod>]
    member __.TestPyramid() =      

        "..\..\..\..\..\skull.jpg"
        |> loadImageAsSignal 
        |> imageFromSignal 
        |> ((convolve 2 (gauss 0.5)) >> decimate)
        |> ((convolve 2 (gauss 0.5)) >> decimate)
        |> ((convolve 2 (gauss 0.5)) >> decimate)
        |> asciiImage 60 
        |> function
            level3 -> (level3 0 0) >= 0.0
        |> Assert.IsTrue


    [<TestMethod>]
    member __.TestMatchReconstruct() =
        "..\..\..\..\..\teapot.jpg"
        |> loadImageAsSignal
        |> imageFromSignal
        |> ((convolve 2 (gauss 0.5)) >> decimate)
        |> ((convolve 2 (gauss 0.5)) >> decimate)
        |> function
            downsampled -> 
                matchReconstruct (200/4) downsampled
        |> asciiImage 60
        |> function _ -> true
        |> Assert.IsTrue