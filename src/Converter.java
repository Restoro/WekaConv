import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.concurrent.ExecutorService;

public class Converter {
	String pathToTrain;
	String pathToData;
	String pathToOutput;
	int dataLineCount;
	int randomNumberCount;

	public Converter(String pathToTrain, String pathToData, String pathToOutput, int randomNumberCount, int dataLineCount) {
		this.pathToTrain = pathToTrain;
		this.pathToData = pathToData;
		this.pathToOutput = pathToOutput;
		this.dataLineCount = dataLineCount;
		this.randomNumberCount = randomNumberCount;
	}
	
	public void executeInThread(ExecutorService exe, String pathToDataSelect, String pathToData,
			String pathToOutput, boolean selectRandom) {
		exe.execute(new Thread(() -> {
			writeArffFile(pathToDataSelect, pathToData, pathToOutput, selectRandom);
		}));
	}

	private void writeIntoArffFile(boolean header, String line, BufferedWriter writer, String pathToData,
			boolean selectRandom) throws IOException {
		BufferedReader arffReader = new BufferedReader(new FileReader(pathToData + line + ".arff"));
		String arffLine;
		while ((arffLine = arffReader.readLine()) != null) {
			if (header)
				writer.write(arffLine + "\n");
			if (arffLine.contains("@data")) {
				break;
			}
		}
		int randomCounter = 1;
		Random r = new Random();
		int randomSelector = r.nextInt(dataLineCount / randomNumberCount) + 1;
		while ((arffLine = arffReader.readLine()) != null) {
			if (!selectRandom) {
				writer.write(arffLine + "\n");
			} else if (randomCounter % randomSelector == 0) {
				writer.write(arffLine + "\n");
				randomSelector = r.nextInt(dataLineCount / randomNumberCount) + 1;
				randomCounter = 1;
			} else {
				randomCounter++;
			}
		}
		arffReader.close();
	}

	private void writeFullArffFile(String pathToData, String pathToOutput) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(pathToOutput, true));
			File folder = new File(pathToData);
			int count = 0;
			for (File file : folder.listFiles()) {
				if (file.isFile()) {
					String name = file.getName().substring(0, file.getName().indexOf('.'));
					writeIntoArffFile(count == 0 ? true : false, name, writer, pathToData, false);
					count++;
				}
			}
			writer.flush();
			writer.close();
			System.out.println("Done writing combined file");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void writeArffFile(String pathToDataSelect, String pathToData, String pathToOutput,
			boolean selectRandom) {
		try {
			Files.deleteIfExists(Paths.get(pathToOutput));
			BufferedReader reader = new BufferedReader(new FileReader(pathToDataSelect));
			BufferedWriter writer = new BufferedWriter(new FileWriter(pathToOutput, true));
			String line;
			int count = 0;
			while ((line = reader.readLine()) != null) {
				line = line.substring(6, line.indexOf('.'));
				writeIntoArffFile(count == 0 ? true : false, line, writer, pathToData, selectRandom);
				count++;
			}
			writer.flush();
			writer.close();
			reader.close();
			System.out.println("Done writing combined file");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
