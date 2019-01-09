namespace LeastSquares.UnitTest

open System
open Microsoft.VisualStudio.TestTools.UnitTesting

open LeastSquaresLib.Helper
open LeastSquaresLib.AsciiGraph
open LeastSquaresLib.VectorND
open LeastSquaresLib.LeastSqOptimizer
open LeastSquaresLib.ImageFilters
open LeastSquaresLib.ImageVector
open LeastSquaresLib.ImageOptimization
open LeastSquaresLib.ImageIO



[<TestClass>]
type TestImageFilters() =

    let dumpImage range image =
        image
        |> asciiImage (seq{0..range})
        |> Seq.iter (printfn "%s")
        image

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
        |> expand |> dumpImage 20
        |> expand |> dumpImage 20
        |> expand |> dumpImage 20
        |> decimate |> dumpImage 20
        |> decimate |> dumpImage 20
        |> decimate |> dumpImage 20
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
        |> imageVectorFromFunc 10
        |> imageFuncFromVector 10
        |> ((convolve 2 (gauss 1.0)) >> decimate)
        |> imageVectorFromFunc 5
        |> imageFuncFromVector 5
        |> ((convolve 2 (gauss 1.0)) >> decimate)
        |> imageVectorFromFunc 3
        |> imageFuncFromVector 3
        |> dumpImage 10
        |> function
            level3 -> (level3 0 0) >= 0.0
        |> Assert.IsTrue

    [<TestMethod>]
    member __.TestMatchReconstruct() =
        circle 5
        |> (convolve 2 (gauss 1.0))
        |> shift -5 -5
        |> matchReconstruct 10
        |> dumpImage 60
        |> function _ -> true
        |> Assert.IsTrue


    // Gaussian Pyramid match reconstruction
    //      compare image reconstructed with decimated Gaussian
    //      to direct Gaussian filter of image (with same kernel)

    // Sparse reconstruction from filter bank
    //      for gaussian filter bank, random select single
    //      pixels and set to 1.0