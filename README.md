# Digitor

Digitor je neuronová síť.

## Instalace

```sh
git clone https://github.com/gyarab/2023-4e-salat-digitor.git
cd 2023-4e-salat-digitor.git/
sh install.sh
```

## Spuštění

```sh
cd build
./digitor [flags] [arguments]
```

#### Možnosti:

* `[flags]`
    * `-t`: Program se spustí v módu učení
    * `-n`: Program místo načítání dat z již existujícího souboru vytvoří nový
* `[arguments]`
    * Použití:
        * `./digitor -t -n <neurony> <iterace> <rychlost_učení> <aktivace> <počet_dat>`
        * `./digitor -t <jméno_soubor> <počet_iterací> <rychlost_učení> <počet_dat>`
        * `./digitor -n <neurony> <aktivace>`
        * `./digitor <jméno_souboru>`
    * Příklady:
        * `./digitor -t -n "784,16,16,10" 1000 0.001 sigmoid 10`
        * `./digitor neuralnetwork.json`
        
