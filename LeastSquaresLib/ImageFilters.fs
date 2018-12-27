namespace LeastSquaresLib

module ImageFilters =

    open VectorND

    type ImageFunc = int->int->float

    [<StructuredFormatDisplay("Ascii")>]
    type SignalImage =
        {   width:int; 
            signal:VectorND 
        } with
        member this.ImageFunc x y =
            let index = y*this.width + x
            if 0 < x && x < this.width
                && 0 < index && index < this.signal.values.Length
            then this.signal.[index]
            else 0.0
        member this.Ascii =
            let range1d = seq {0..this.width}
            let values = 
                (range1d, range1d) 
                ||> Seq.allPairs
                |> Seq.map (fun (x,y) -> this.ImageFunc x y)
            let (minValue, maxValue) = 
                (values |> Seq.min), (values |> Seq.max)
            let asciiPixelArray = [|"   "; " . "; " .."; "..."; "..:"; ".::"; ":::"|]
            let index value = 
                float (asciiPixelArray.Length-1)
                    * (value - minValue) / (maxValue - minValue + 1.0)
            let asciiPixel value = 
                asciiPixelArray.[int (index value)]
            range1d
            |> Seq.map (fun row -> 
                    System.String.Join("", 
                        range1d |> Seq.map (fun column -> asciiPixel(this.ImageFunc column row))))
            |> Seq.concat     

    //(* helper to expose a signal as an image *)
    //let imageFromSignal (signalImage:SignalImage) =
    //    signalImage.ImageFunc

    let imageToSignal width imageFunc =
        { width = width;
            signal = 
                (seq {0..width-1}, seq {0..width-1})
                ||> Seq.allPairs
                |> Seq.map (fun (x,y) -> imageFunc x y)
                |> Array.ofSeq
                |> VectorND }        


    (* create rectangle function *)
    let rectangle width x y = 
        if -width<x && x<width && -width<y && y<width then 1.0 else 0.0

    (* create circle function *)
    let circle radius x y = 
        if (x*x + y*y) < radius*radius then 1.0 else 0.0

    (* create gauss function *)
    let gauss sigma x y = exp(float -(x*x + y*y) / (float sigma * sigma))

    (* create gabor function *)
    let gabor sigma kx ky x y = 
        (gauss sigma x y) * cos((float x)*kx + (float y)*ky)

    (* create parabolic function *)
    let parab x y = float (x*x + y*y)

    (* decimate operator *)
    let decimate image x y = image (x*2) (y*2)

    (* expand operator *)
    let expand image x y = image (x/2) (y/2)
        
    (* shift operator *)
    let shift sx sy imageFunc x y = imageFunc (x+sx) (y+sy)

    (* convolve operator *)
    let convolve kSize (kernel:ImageFunc) (image:ImageFunc) x y =
        (seq {-kSize..kSize}, seq {-kSize..kSize})
        ||> Seq.allPairs
        |> Seq.map (fun (kx,ky) -> (kernel kx ky) * (image (x+kx) (y+ky)))
        |> Seq.sum
