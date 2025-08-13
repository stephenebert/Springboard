import Foundation

public struct TagResult: Identifiable, Hashable {
    public let id = UUID()
    public let label: String
    public let confidence: Double

    public init(label: String, confidence: Double) {
        self.label = label
        self.confidence = confidence
    }
}
