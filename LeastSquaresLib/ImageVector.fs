namespace LeastSquaresLib

module ImageVector =

    open System.Diagnostics
    open VectorND
    open LeastSquaresLib
    open LeastSquaresLib.ImageFilters

    type IndexBounds = 
        {xmin:int; ymin:int; xmax:int; ymax:int}
        member this.allIndices =
            (seq {this.xmin..this.xmax}, 
                    seq {this.ymin..this.ymax})
            ||> Seq.allPairs

    let imageVectorFromFuncBounds (bounds:IndexBounds) (imageFunc:ImageFunc) =
        bounds.allIndices
        |> Seq.map (fun (x,y) -> imageFunc x y)
        |> Array.ofSeq
        |> VectorND   

    let imageFuncFromVectorBounds (b:IndexBounds) (v:VectorND) x_in y_in =
        let (w, h) = (b.xmax - b.xmin, b.ymax - b.ymin)
        let (x, y) = (x_in - b.xmin, y_in - b.ymin)
        y * w + x
        |> function 
            | index when 0 < x && x < w -> v.[index]
            | index when 0 < y && y < h -> v.[index]
            | _ -> 0.0
        
    let imageVectorFromFunc width (imageFunc:ImageFunc) =
        (seq {0..width-1}, seq {0..width-1})
        ||> Seq.allPairs
        |> Seq.map (fun (x,y) -> imageFunc x y)
        |> Array.ofSeq
        |> VectorND


    let imageFuncFromVector width (v:VectorND) x y =
        let index = y*width + x
        if 0 < x && x < width
            && 0 < index && index < v.values.Length
        then v.[index]
        else 0.0

    [<StructuredFormatDisplay("Ascii")>]
    type ImageVector(width:int, b:IndexBounds, vector:VectorND) =        
        do Trace.Assert(vector.values.Length = width * width)
        member this.Width = width
        member this.Vector = vector
        member this.Ascii =
            imageFuncFromVector this.Width this.Vector
            |> AsciiGraph.asciiImage (seq {0..this.Width}) 
            |> String.concat "\n"

    // pyramid from an ImageVector
    type PyramidVector(levels:int, width0:int, vector:VectorND) =
        do Trace.Assert(vector.values.Length = width0 * width0)
        member this.Width0 = width0
        member this.Vector = vector
        member this.levels = levels
        member this.ImageFunc n x y =
            0.0

    let createPyramidLevel (iv:ImageVector) =
        (iv.Width/2,
            {xmin=0; xmax=iv.Width/2; ymin=0; ymax=0},
            iv.Vector
            |> imageFuncFromVector iv.Width
            |> convolve 3 (gauss 2.0)
            |> decimate
            |> imageVectorFromFunc (iv.Width/2))
        |> ImageVector

    let levelFromPyramid level width0 v =
        width0*width0