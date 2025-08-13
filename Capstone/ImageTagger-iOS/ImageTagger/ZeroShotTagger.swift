import Foundation
import Vision
import CoreML
import UIKit

/// Central, testable classifier with:
/// 1) Vision’s built-in classifier
/// 2) Automatic fallback to bundled MobileNetV2 if Vision fails
final class ZeroShotTagger {
    static let shared = ZeroShotTagger()
    private init() {}

    // MARK: - Public API

    /// Classify an image and return de-duplicated tags.
    /// - Parameters:
    ///   - uiImage: source image
    ///   - topK: cap result count (applied after filtering)
    ///   - threshold: minimum confidence (0...1)
    ///   - focusTerms: optional “soft boost” terms (lowercased)
    ///   - applyBoost: if true, add +0.15 confidence for focus term hits
    func predictTags(
        from uiImage: UIImage,
        topK: Int,
        threshold: Double,
        focusTerms: [String],
        applyBoost: Bool
    ) throws -> [TagResult] {

        // Autorelease & downscale to avoid Espresso context failures
        return try autoreleasepool {
            let cg = try uiImage.forceCGImageResized(maxDimension: 1024)

            // Try Vision first
            do {
                let results = try classifyWithVision(cgImage: cg)
                return postProcess(results,
                                   topK: topK,
                                   threshold: threshold,
                                   focusTerms: focusTerms,
                                   applyBoost: applyBoost)
            } catch {
                // Fallback to MobileNetV2 Core ML
                let results = try classifyWithMobileNet(cgImage: cg)
                return postProcess(results,
                                   topK: topK,
                                   threshold: threshold,
                                   focusTerms: focusTerms,
                                   applyBoost: applyBoost)
            }
        }
    }

    // MARK: - Inference backends

    private func classifyWithVision(cgImage: CGImage) throws -> [TagResult] {
        let request = VNClassifyImageRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let observations = request.results as? [VNClassificationObservation],
              !observations.isEmpty else {
            throw NSError(domain: "ZeroShotTagger",
                          code: 101,
                          userInfo: [NSLocalizedDescriptionKey: "No Vision results"])
        }

        return observations
            .sorted { $0.confidence > $1.confidence }
            .flatMap { obs -> [TagResult] in
                obs.identifier
                    .replacingOccurrences(of: "_", with: " ")
                    .components(separatedBy: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                    .map { TagResult(label: $0, confidence: Double(obs.confidence)) }
            }
    }

    private func classifyWithMobileNet(cgImage: CGImage) throws -> [TagResult] {
        guard let url = findCompiledMobileNet() else {
            throw NSError(domain: "ZeroShotTagger",
                          code: 102,
                          userInfo: [NSLocalizedDescriptionKey: "MobileNetV2 model not found"])
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndGPU

        let model = try MLModel(contentsOf: url, configuration: cfg)
        let vnModel = try VNCoreMLModel(for: model)

        let req = VNCoreMLRequest(model: vnModel)
        req.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([req])

        guard let observations = req.results as? [VNClassificationObservation],
              !observations.isEmpty else {
            throw NSError(domain: "ZeroShotTagger",
                          code: 103,
                          userInfo: [NSLocalizedDescriptionKey: "No Core ML results"])
        }

        return observations
            .sorted { $0.confidence > $1.confidence }
            .flatMap { obs -> [TagResult] in
                obs.identifier
                    .replacingOccurrences(of: "_", with: " ")
                    .components(separatedBy: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                    .map { TagResult(label: $0, confidence: Double(obs.confidence)) }
            }
    }

    // MARK: - Post-processing

    private func postProcess(
        _ raw: [TagResult],
        topK: Int,
        threshold: Double,
        focusTerms: [String],
        applyBoost: Bool
    ) -> [TagResult] {

        // De-duplicate by label (case-insensitive), keeping max confidence
        var dedup: [String: TagResult] = [:]
        for t in raw {
            let key = t.label.lowercased()
            if let existing = dedup[key] {
                if t.confidence > existing.confidence { dedup[key] = t }
            } else {
                dedup[key] = t
            }
        }
        var tags = Array(dedup.values)

        // Optional soft boost for user focus terms
        if applyBoost, !focusTerms.isEmpty {
            let focusSet = Set(focusTerms.map { $0.lowercased() })
            tags = tags.map { t in
                let boosted = focusSet.contains(t.label.lowercased())
                    ? min(1.0, t.confidence + 0.15)
                    : t.confidence
                return TagResult(label: t.label, confidence: boosted)
            }
        }

        // Filter & cap
        return tags
            .filter { $0.confidence >= threshold }
            .sorted { $0.confidence > $1.confidence }
            .prefix(topK)
            .map { $0 }
    }

    // MARK: - Helpers

    private func findCompiledMobileNet() -> URL? {
        // Look for any .mlmodelc (prefer “mobilenet” in name)
        let fm = FileManager.default
        guard let en = fm.enumerator(at: Bundle.main.bundleURL,
                                     includingPropertiesForKeys: nil) else { return nil }
        let all = en.compactMap { $0 as? URL }
            .filter { $0.pathExtension == "mlmodelc" }
        if let exact = all.first(where: { $0.lastPathComponent.lowercased().contains("mobilenetv2") }) {
            return exact
        }
        return all.first
    }
}

// MARK: - UIImage downscale / CGImage helpers
private extension UIImage {
    /// Returns a CGImage, resizing the longest side to `maxDimension` to reduce memory.
    func forceCGImageResized(maxDimension: CGFloat) throws -> CGImage {
        if let cg = self.cgImage {
            // If already small enough, return as-is
            if max(CGFloat(cg.width), CGFloat(cg.height)) <= maxDimension { return cg }
            return try cg.resized(maxDimension: maxDimension)
        }

        let ctx = CIContext(options: nil)
        guard let ci = self.ciImage ?? CIImage(image: self) else {
            throw NSError(domain: "ZeroShotTagger", code: 199,
                          userInfo: [NSLocalizedDescriptionKey: "Could not create CIImage"])
        }
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else {
            throw NSError(domain: "ZeroShotTagger", code: 198,
                          userInfo: [NSLocalizedDescriptionKey: "Could not create CGImage"])
        }
        if max(CGFloat(cg.width), CGFloat(cg.height)) <= maxDimension { return cg }
        return try cg.resized(maxDimension: maxDimension)
    }
}

private extension CGImage {
    func resized(maxDimension: CGFloat) throws -> CGImage {
        let w = CGFloat(self.width)
        let h = CGFloat(self.height)
        let scale = maxDimension / max(w, h)
        let newW = max(1, Int(w * scale))
        let newH = max(1, Int(h * scale))

        guard let colorSpace = self.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB) else {
            throw NSError(domain: "ZeroShotTagger", code: 197,
                          userInfo: [NSLocalizedDescriptionKey: "Missing color space"])
        }

        guard let ctx = CGContext(
            data: nil,
            width: newW,
            height: newH,
            bitsPerComponent: self.bitsPerComponent,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: self.bitmapInfo.rawValue
        ) else {
            throw NSError(domain: "ZeroShotTagger", code: 196,
                          userInfo: [NSLocalizedDescriptionKey: "Resize context failed"])
        }
        ctx.interpolationQuality = .high
        ctx.draw(self, in: CGRect(x: 0, y: 0, width: CGFloat(newW), height: CGFloat(newH)))
        guard let out = ctx.makeImage() else {
            throw NSError(domain: "ZeroShotTagger", code: 195,
                          userInfo: [NSLocalizedDescriptionKey: "Resize output failed"])
        }
        return out
    }
}
