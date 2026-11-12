from pupil_on_pole import experiments as ex

def main():
    out = ex.create_experiment("pilot", seeds=(1,), workers=3)
    ex.generate_dataset(out)                
    ex.train_experiment(out)                 
    ex.evaluate_experiment(out, held_out=False)
    results = ex.load_experiment(out)
    ex.plot_experiment(results, out)
    print(results["curves"][["method", "online_step", "lap_time_s", "finish_rate"]])
    print(f"Figures dans {out / 'figures'}")

if __name__ == "__main__":   
    main()
