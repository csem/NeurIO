import typer
import benchmark as nnb
import sys

app = typer.Typer()


# Benchmark command
@app.command()
def benchmark(
        device: str = typer.Option(
            None,
            "--device",
            "-d",
            prompt="Device?"
        ),
        task: str = typer.Option(
            None,
            "--task",
            "-t",
            prompt="Task?"
        ),
        model: str = typer.Option(
            None,
            "--model",
            "-m",
            prompt="Model ?"
        ),
        compression: str = typer.Option(
            'none',
            "--compression",
            "-c",
        ),
        input_dataset: str = typer.Option(
            None,
            "--input_dataset",
            "-id",
        ),
        output_dataset: str = typer.Option(
            None,
            "--output_dataset",
            "-od",
        ),
        batch_size: int = typer.Option(
            None,
            "--batch_size",
            "-b",
        ),
        is_quantized: bool = typer.Option(
            False,
            "--is_quantized",
            "-q"
        ),
        output: str = typer.Option(
            None,
            "--output",
            "-o",
            prompt="Output?"
        ),
) -> None:
    # Try to run the benchmark function
    try:
        nnb.benchmark(device, task, model, compression, input_dataset,
                      output_dataset, batch_size, is_quantized, output)

        # Print success if succeeded. THIS LINE IS IMPORTANT since it marks the end of the testbench.
        print('testbench done : success')

    # If an exception is caught
    except Exception as e:

        # print the exeption
        print('Oops! An exception occured: {} '.format(str(e)))
        raise e

        # Print failure. THIS LINE IS IMPORTANT since it marks the end of the testbench.
        print('testbench done : failure')


# Device command
@app.command()
def devices() -> None:
    nnb.devices()


# Entry point
if __name__ == "__main__":
    app()
