import time
import matplotlib.pyplot as plt

def process():
	# Una operación cualquiera
	return 2+2



def  rt_main(interval_time = 0.001, iters = 100):
	latencies = []

	# Guardamos el tiempo de comienzo
	start_time = time.time()
	# El comienzo del primer intervalo debería ser igual a start_time
	expected_time = start_time

	for n in range(iters):
		# T1: El intervalo comienza
		# Se calcula la latencia
		# (tiempo verdadero en el que comienza el intervalo - tiempo esperado de comienzo)
		iter_time = time.time()
		latencies.append((iter_time - expected_time) * 1000) # Por 1000 para ponerlo en milisegundos
		expected_time = iter_time + interval_time # Guardamos el tiempo de comienzo esperado para el proximo

		# T2: Realiza alguna acción
		process()

		# Calculamos el tiempo que ya ha transcurrido del intervalo
		process_time = time.time() - iter_time


		# T3: Duerme hasta que toque 
		# el siguiente intervalo
		# (duracion total del intervalo - tiempo ya transcurrido del mismo)
		# Por si acaso process_time es mayor que interval_time hacemos el maximo con 0
		sleep_time = max(interval_time - process_time, 0)
		time.sleep(sleep_time)


	# Guardamos el tiempo de finalización
	end_time = time.time()

	print("Expected time: ", iters * interval_time)
	print("Execution time: ", end_time - start_time)

	# Hacemos el histograma de latencias con 100 barras
	# En el eje X se representan los valores de latencias
	# y en el Y cuantas veces se ha dado ese valor
	plt.hist(latencies, 100)
	plt.ylabel("Occurrences")
	plt.xlabel("Latency (ms)")
	plt.show()
