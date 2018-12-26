namespace LeastSquaresLib

module ImageFilters =

    type ImageFunc = int->int->float

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
