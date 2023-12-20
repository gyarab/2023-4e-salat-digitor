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
    * Příklad: `./digitor -t <jmeno_soubor> <počet_iterací> <rychlost_učení> <počet_dat>`
