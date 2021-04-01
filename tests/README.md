# WILDS Smoke Tests

## Prerequisite

Install the CodaLab CLI and other dependencies, by running: 
    
    pip install -r requirements.dev.txt

Make sure the CodaLab account you will use to run the tests is part of the `wilds-admins` group on CodaLab.

## Testing WILDS

1. Switch to the [test worksheet](https://worksheets.codalab.org/worksheets/0xacd40b3e4991410b98643b9cc0b10347) 
by running `cl work 0xacd40b3e4991410b98643b9cc0b10347`.
2. Upload the WILDS source to test: `cl upload <path to WILDS source to test>`.
3. Run the smoke test script: 

    `python3 tests/smoke_test.py --wilds-src-uuid <UUID of the WILDS source from step 2>` 

    This will create run bundles on the test worksheet. After the script runs, it will output a unique ID which 
    you will need for the evaluation step. 
4. Once all the run finishes (it will take around 2 hours), evaluate the test runs:
    
    `python3 tests/smoke_test.py --run-id <unique ID from step 3>` 

    This command will output a results JSON file `results_<unique ID from step 3>.json`.
5. Compare the results of `results_<unique ID from step 3>.json` to the previous results JSON. 
6. If the new results look reasonable compared to the previous results, delete the old file. 
Then, commit and push the new results file to the [GitHub repository](https://github.com/p-lambda/wilds). 
