namespace LeastSquaresLib

module ImageFilters =

    open LeastSquaresLib.VectorND
    open SixLabors.ImageSharp

    type ImageFunc = int->int->float

    type ImageAsSignal = { width:int; signal:VectorND }

    (* helper to expose a signal as an image *)
    let imageFromSignal { width=width; signal=signal } x y =
        let index = y*width + x
        if 0 < x && x < width
            && 0 < index && index < signal.values.Length
        then signal.[index]
        else 0.0

    (* loads an image from file, returning the signal (vector) and
        width representations *)
    let loadImageAsSignal (fileName:string) =
        let loaded = Image.Load(fileName)
        let width, height = loaded.Width, loaded.Height
        (seq {0..width-1}, seq {0..height-1})
        ||> Seq.allPairs
        |> Seq.map 
            (fun (x,y) 
                -> (float loaded.[y,x].G) / 255.0)
        |> Array.ofSeq
        |> VectorND
        |> function 
            signal -> { width=width; signal=signal }
        
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
    let decimate image = fun x y -> (image (x*2) (y*2))

    (* expand operator *)
    let expand image = fun x y -> (image (x/2) (y/2))

    (* convolve operator *)
    let convolve kSize (kernel:ImageFunc) (image:ImageFunc) x y =
        (seq {-kSize..kSize}, seq {-kSize..kSize})
        ||> Seq.allPairs
        |> Seq.map (fun (kx,ky) -> (kernel kx ky) * (image (x+kx) (y+ky)))
        |> Seq.sum

    (* ascii image output *)
    let asciiImage range (image:ImageFunc) =
        let range1d = seq {0..range}
        let values = 
            (range1d, range1d) 
            ||> Seq.allPairs
            |> Seq.map (fun (x,y) -> image x y)
        let min, max = values |> Seq.min, values |> Seq.max
        let asciiPixelArray = [|"   "; " . "; " .."; "..."; "..:"; ".::"; ":::"|]
        let index value = (value - min) / (max - min) * float (asciiPixelArray.Length-1)
        let asciiPixel value = 
            asciiPixelArray.[int (index value)]
        range1d
        |> Seq.map (fun row -> 
                System.String.Join("", 
                    range1d |> Seq.map (fun column -> asciiPixel(image column row))))
        |> Seq.iter (printfn "%s")