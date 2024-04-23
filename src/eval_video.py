
from compress.training.video.eval import *
from compress.training.video.utils import *



def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args) #dddd

    #if not args.source:
    #    print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
    #    parser.print_help()
    #    raise SystemExit(1)
    description = ("entropy-estimation" if args.entropy_estimation else args.entropy_coder) #ssss
    filepaths = collect_videos(args.dataset)
    if len(filepaths) == 0:
        print("Error: no video found in directory.", file=sys.stderr)
        raise SystemExit(1)

    # create output directory
    outputdir = args.output
    Path(outputdir).mkdir(parents=True, exist_ok=True)

    print("output dir è: ",outputdir)


    if args.source == "pretrained":
        args.qualities = [int(q) for q in args.quality.split(",") if q]
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d}"
    else:
        args.qualities = [int(q) for q in args.quality.split(",") if q]
        runs = create_runs(args.path,args.qualities)
        print("entro qua: ",runs)
        opts = (args.architecture, args.no_update) #dddd
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s}"


    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        print("the run is: ",run)
        model = load_func(*opts, run)
        if args.source == "pretrained":
            trained_net = f"{args.architecture}-{args.metric}-{run}-{description}"
        else:
            quality = run.split("/")[-2]
            print("the quality is: ",quality,"  ",description)
            cpt_name = f"{args.architecture}-{args.metric}-{quality}" # removesuffix() python3.9
            trained_net = f"{cpt_name}-{description}"
        #print(f"Using trained model {trained_net}", file=sys.stderr)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
            if args.half:
                model = model.half()
        args_dict = vars(args)
        print(f"Using trained model {trained_net}", file=sys.stderr)
        
        metrics = run_inference(
            filepaths,
            args.dataset,
            model,
            outputdir,
            trained_net=trained_net,
            description=description,
            **args_dict,
        )
        results["q"].append(trained_net)

        
        for k, v in metrics.items():
            results[k].append(v)

        
    
    output = {
        "name": f"{args.architecture}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }

    print("the final output is: ",output) #sssssssss

    if args.output_file == "":
        output_file = f"{args.output_path}/{args.architecture}-{description}"
    else:
        output_file = args.output_file

    with (Path(f"{outputdir}/{output_file}").with_suffix(".json")).open("wb") as f:
        f.write(json.dumps(output, indent=2).encode())
    print(json.dumps(output, indent=2))
    

    


if __name__ == "__main__":
    main(sys.argv[1:])